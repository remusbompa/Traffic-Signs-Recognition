from __future__ import division
import time

from CVModule.DrawImages import ImagesHandler
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images',
                        help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det',
                        help="Image / Directory to store detections to",
                        default="det", type=str)
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
                        help="Input resolution of the network. "
                             "Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

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

    read_dir = time.time()
    print_info(widget, False, "info", "Reading addresses.....")
    images = args.images
    im_list = []
    img = None
    try:
        for img in images:
            if os.path.isabs(img):
                im_list.append(img)
            else:
                im_list.append(osp.join(osp.realpath('.'), img))
    except FileNotFoundError:
        print_info(widget, True, "error", "No file or directory with the name {}".format(img))

    if not os.path.exists(args.det):
        os.makedirs(args.det)
    print_info(widget, False, "info", "Finished reading addresses")
    finish_read_dir = time.time()

    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    namesfile = args.names

    cuda_present = torch.cuda.is_available()

    classes = load_classes(namesfile)
    num_classes = len(classes)

    # Set up the neural network
    load_net = time.time()
    print_info(widget, False, "info", "Loading network.....")
    model = Darknet(args.cfg)
    model.load_weights(args.weights)
    print_info(widget, False, "info", "Network successfully loaded")
    finish_load_net = time.time()

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if cuda_present:
        model.cuda()
    # Set the model in evaluation mode (for Batchnorm layers)
    model.eval()
    # Detection phase

    load_batch = time.time()
    print_info(widget, False, "info", "Loading batches.....")
    loaded_ims = [cv2.imread(x) for x in im_list]

    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(im_list))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(im_list) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in range(num_batches)]

    write = 0

    if cuda_present:
        im_dim_list = im_dim_list.cuda()
    print_info(widget, False, "info", "Finished loading batches....")
    start_det_loop = time.time()
    for i, batch in enumerate(im_batches):
        # load the image
        start = time.time()
        print_info(widget, False, "info", f"Detecting batch no {i}....")
        if cuda_present:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), cuda_present)

        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

        end = time.time()

        if type(prediction) == int:

            for im_num, image in enumerate(im_list[i * batch_size: min((i + 1) * batch_size, len(im_list))]):
                im_id = i * batch_size + im_num
                msg = "{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size)
                msg += "\n{0:20s} {1:s}".format("Objects Detected:", "")
                msg += "\n----------------------------------------------------------"
                print_info(widget, False, 'batch_info', msg, im_id)
            continue

        prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in imlist

        if not write:  # If we have't initialised output
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(im_list[i * batch_size: min((i + 1) * batch_size, len(im_list))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            msg = "{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size)
            msg += "\n{0:20s} {1:s}".format("Objects Detected:", " ".join(objs))
            msg += "\n----------------------------------------------------------"
            print_info(widget, False, 'batch_info', msg, im_id)

        if cuda_present:
            torch.cuda.synchronize()
        print_info(widget, False, "info", f"Finished detecting batch no {i}")
    try:
        output
    except NameError:
        print_info(widget, False, 'no_detections', "No detections were made")
        print_info(widget, False, 'finished')
        return

    ## Start rescaling
    print_info(widget, False, "info", "Output processing....")
    output_rescale = time.time()
    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])
    class_load = time.time()
    print_info(widget, False, "info", "Finished output processing.")

    # Start draw
    print_info(widget, False, "info", "Drawing boxes....")
    draw = time.time()
    images_handler = ImagesHandler(classes, output, loaded_ims, args.det, im_list, batch_size)
    images_handler.write()
    print_info(widget, False, "images_ready", images_handler.imageList)
    end = time.time()
    print_info(widget, False, "info", "Finished drawing boxes")

    msg = "\n\nSUMMARY"
    msg += "\n----------------------------------------------------------"
    msg += "\n{:25s}: {}".format("Task", "Time Taken (in seconds)")
    msg += "\n"
    msg += "\n{:25s}: {:2.3f}".format("Reading addresses", finish_read_dir - read_dir)
    msg += "\n{:25s}: {:2.3f}".format("Loading network", finish_load_net - load_net)
    msg += "\n{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch)
    msg += "\n{:25s}: {:2.3f}".format("Detection (" + str(len(im_list)) + " images)", output_rescale - start_det_loop)
    msg += "\n{:25s}: {:2.3f}".format("Output Processing", class_load - output_rescale)
    msg += "\n{:25s}: {:2.3f}".format("Drawing Boxes", end - draw)
    msg += "\n{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(im_list))
    msg += "\n----------------------------------------------------------"
    print_info(widget, False, 'info', msg)
    torch.cuda.empty_cache()

    print_info(widget, False, 'finished')


if __name__ == '__main__':
    main()