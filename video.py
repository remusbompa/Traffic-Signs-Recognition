import mimetypes
import os
import time

from PyQt5.QtCore import QThread

from CVModule.DrawImages import ImageHandler
from util import *
import argparse
from darknet import Darknet


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfg', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weights', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--names", dest='names', help="namesfile",
                        default="data/coco.names", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. Increase to increase accuracy."
                             " Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--video", dest="video", help="Video file to run detection on", default="video.avi",
                        type=str)

    return parser.parse_args()


def print_info(widget, error, signal, *msg):
    if not widget:
        print(msg)
        if error:
            exit()
    else:
        signal = getattr(widget.obj, signal)
        signal.emit(*msg)
        if error:
            signal.emit("finished")


def main(widget=None):
    if not widget:
        args = arg_parse()
    else:
        args = widget.args
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    names_file = args.names
    cuda = torch.cuda.is_available()

    classes = load_classes(names_file)
    num_classes = len(classes)

    # Check files
    if args.video != 0:
        if not os.path.exists(args.video):
            print_info(widget, True, "error", "No file with the name {}".format(args.video))
        mimetypes.init()
        mime_start = mimetypes.guess_type(args.video)[0]
        if mime_start:
            mime_start = mime_start.split('/')[0]
            if mime_start != 'video':
                print_info(widget, True, "error", "No video file with the name {}".format(args.video))

    if not os.path.exists(args.det):
        os.makedirs(args.det)
    # Set up the neural network
    print_info(widget, False, "info", "Loading network.....", -1)
    model = Darknet(args.cfg)
    model.load_weights(args.weights)
    print_info(widget, False, "info", "Network successfully loaded", 0)

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if cuda:
        model.cuda()
    # Set the model in evaluation mode
    model.eval()

    # Detection phase
    video_file = args.video  # or path to the video file.
    if type(video_file) == int:
        cap = cv2.VideoCapture(0)  # for webcam
        if not cap:
            print_info(widget, True, "error", "Error opening web cam")
    else:
        cap = cv2.VideoCapture(video_file)
        if not cap:
            print_info(widget, True, "error", f"Error opening video file {video_file}")

    assert cap.isOpened(), 'Cannot capture source'

    frames = 0
    start = time.time()
    print_info(widget, False, "info", "Began capturing", -2)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if widget.obj.cancel:
                current_time = time.time()
                print_info(widget, False, "info", "Canceled processing", current_time - start)
                widget.obj.cancel = False
                QThread.currentThread().quit()
                return
            widget.obj.pauseMutex.lock()
            if widget.obj.pause:
                widget.obj.pauseCond.wait(widget.obj.pauseMutex)
            widget.obj.pauseMutex.unlock()

            img = prep_image(frame, inp_dim)

            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

            if cuda:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img, volatile=True), cuda)
            output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                current_time = time.time()
                msg = " FPS of the video is {:5.4f}".format(frames / (current_time - start))
                print_info(widget, False, "info", msg, current_time - start)

                image_handler = ImageHandler(0, 0, f"frame{frames}", frame)
                print_info(widget, False, "image_ready", image_handler)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            image_handler = ImageHandler(0, 0, f"frame{frames}", frame)
            image_handler.write(output, classes)
            print_info(widget, False, "image_ready", image_handler)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            frames += 1
            current_time = time.time()
            msg = " FPS of the video is {:5.2f}".format(frames / (current_time - start))
            print_info(widget, False, "info", msg, current_time - start)

        else:
            break


if __name__ == '__main__':
    main()
