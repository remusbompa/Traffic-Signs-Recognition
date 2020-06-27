import mimetypes
import os
import time
from queue import Queue
from threading import Thread
from PyQt5.QtCore import QThread
from DeepSort.deepsort import DeepSort
from VisionPackage.DrawImages import ImageHandler
from tracking import Sort
from utilities import *
import argparse
from model import Darknet


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


def detect_video(widget=None, with_tracking=None):
    if not widget:
        args = arg_parse()
    else:
        args = widget.args
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

    # Set up the neural network
    # Detection phase
    video_file = args.video  # or path to the video file.
    if type(video_file) == int:
        # for webcam
        reading_thread = ReadFramesThread(0, args, with_tracking, widget).start()
    else:
        reading_thread = ReadFramesThread(video_file, args, with_tracking, widget).start()

    while reading_thread.has_images() or (not reading_thread.stopped):
        if widget.obj.cancel:
            reading_thread.cancel()
            widget.obj.cancel = False
            QThread.currentThread().quit()
            return

        if widget.obj.pause:
            reading_thread.pause()
            widget.obj.pauseMutex.lock()
            widget.obj.pauseCond.wait(widget.obj.pauseMutex)
            widget.obj.pauseMutex.unlock()

        image_handler = reading_thread.get_image()
        last_contour = image_handler.get_last_contour()
        if last_contour:
            print_info(widget, False, "contour_ready", last_contour)
        print_info(widget, False, "image_ready", image_handler)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break


class ReadFramesThread:
    def __init__(self, path, args, with_tracking, widget, queue_size=3000):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.widget = widget
        self.tracking = with_tracking
        if not self.stream:
            if type(path) == int:
                print_info(widget, True, "error", f"Error opening web cam on {path}")
            else:
                print_info(widget, True, "error", f"Error opening video file {path}")
        self.stopped = False
        self.canceled = False
        self.paused = False
        self.ready = False
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        self.imread = Queue(maxsize=queue_size)
        self.Q_processed = Queue(maxsize=queue_size)

        self.inp_dim = int(args.reso)
        self.batch_size = int(args.bs)
        self.names_file = args.names
        self.confidence = float(args.confidence)
        self.nms_thresh = float(args.nms_thresh)
        self.is_classifier = args.is_classifier
        self.classes = load_classes(self.names_file)
        self.num_classes = len(self.classes)

        self.model = None
        self.model_classifier = None
        if self.is_classifier:
            print_info(widget, False, "info", "Loading network for detection.....", -1)
            self.model = Darknet(args.classifier_cfg)
            self.model.load_weights(args.classifier_weights)
            print_info(widget, False, "info", "Network for detection successfully loaded", 0)

            print_info(widget, False, "info", "Loading network for classification.....", -1)
            self.model_classifier = Darknet(args.cfg)
            self.model_classifier.load_weights(args.weights)
            print_info(widget, False, "info", "Network for classification successfully loaded", 0)

            self.model_classifier.net_info["height"] = args.reso
            self.inp_dim = int(self.model_classifier.net_info["height"])
            # If there's a GPU availible, put the model on GPU
            self.cuda = torch.cuda.is_available()
            if self.cuda:
                self.model_classifier.cuda()
            # Set the model in evaluation mode
            self.model_classifier.eval()

            self.classifier_confidence = self.confidence
            self.classifier_nms_thesh = self.nms_thresh
            self.classifier_classes = self.classes
            self.classifier_num_classes = self.num_classes
            self.classifier_names_file = self.names_file
            self.classifier_inp_dim = self.inp_dim

            self.inp_dim = args.classifier_inp_dim
            self.confidence = args.classifier_confidence
            self.nms_thresh = args.classifier_nms_thresh
            self.names_file = args.classifier_names
            self.classes = load_classes(self.names_file)
            self.num_classes = len(self.classes)

        else:
            print_info(widget, False, "info", "Loading network.....", -1)
            self.model = Darknet(args.cfg)
            self.model.load_weights(args.weights)
            print_info(widget, False, "info", "Network successfully loaded", 0)

        self.model.net_info["height"] = self.inp_dim
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        # If there's a GPU availible, put the model on GPU
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()
        # Set the model in evaluation mode
        self.model.eval()

        # if tracking selected, initialize sort class
        self.mot_tracking = None
        if self.tracking == "sort":
            self.mot_tracking = Sort(max_age=30, min_hits=3)
        elif self.tracking == "deep_sort":
            print_info(widget, False, "info", "Loading Deep Sort model ...", -1)
            self.mot_tracking = DeepSort()
            print_info(widget, False, "info", "Deep Sort model loaded", -1)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        # t.daemon = True
        t.start()
        return self

    def update(self):
        frames = 0
        start = time.time()
        print_info(self.widget, False, "info", "Began capturing", -2)
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break
            if self.canceled:
                current_time = time.time()
                print_info(self.widget, False, "info", "Canceled processing", current_time - start)
                return
            if self.paused:
                self.widget.obj.pauseMutex.lock()
                self.widget.obj.pauseCond.wait(self.widget.obj.pauseMutex)
                self.widget.obj.pauseMutex.unlock()
                self.paused = False
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    self.ready = True
                    return
                # add the frame to the queue
                self.Q.put(prep_image(frame, self.inp_dim))
                self.imread.put(frame)

                frames += 1
                current_time = time.time()
                msg = " FPS of the video is {:5.4f}".format(frames / (current_time - start))
                print_info(self.widget, False, "info", msg, current_time - start)

                if frames % self.batch_size == 0:
                    self.process_frames()
        if not self.Q.empty():
            self.process_frames()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def cancel(self):
        self.canceled = True

    def pause(self):
        self.paused = True

    def has_batch(self):
        if self.Q.qsize() >= self.batch_size:
            return True
        if self.Q.qsize() > 0 and self.stopped:
            return True
        return False

    def get_batch(self):
        if (self.Q.qsize() >= self.batch_size) or (self.Q.qsize() > 0 and self.stopped):
            res = np.empty((0, 0))
            im_dim_list = []
            imread_list = []
            for _ in range(self.batch_size):
                img = self.Q.get()
                if np.size(res, 0) == 0:
                    res = img
                else:
                    res = torch.cat((res, img))
                img = self.imread.get()
                im_dim_list.append((img.shape[1], img.shape[0]))
                imread_list.append(img)
            im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
            return res, im_dim_list, imread_list
        return False, False, False

    def process_frames(self):
        batch_nr = -1
        batch, im_dims, imread = self.get_batch()
        if imread:
            batch_nr += 1
            if self.cuda:
                im_dims = im_dims.cuda()
                batch = batch.cuda()
            with torch.no_grad():
                output = self.model(batch, self.cuda)

            for frame_id in range(np.size(output, 0)):
                nr_frame = self.batch_size * batch_nr + frame_id + 1
                im_dim = im_dims[frame_id]
                frame = output[frame_id].unsqueeze(0)
                frame = write_results(frame, self.confidence, self.num_classes, nms_conf=self.nms_thresh)

                if np.size(frame, 0) > 0:
                    im_dim = im_dim.repeat(frame.size(0), 1)
                    scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

                    frame[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
                    frame[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

                    frame[:, 1:5] /= scaling_factor

                    for i in range(frame.shape[0]):
                        frame[i, [1, 3]] = torch.clamp(frame[i, [1, 3]], 0.0, im_dim[i, 0])
                        frame[i, [2, 4]] = torch.clamp(frame[i, [2, 4]], 0.0, im_dim[i, 1])

                    if self.is_classifier:
                        frame = self.apply_classifier_model(imread[frame_id], frame)

                if self.tracking == "sort":
                    if self.cuda:
                        frame = frame.cpu()
                    frame = self.mot_tracking.update(frame)
                    if self.cuda:
                        frame = torch.from_numpy(frame).cuda()
                elif self.tracking == "deep_sort":
                    if self.cuda:
                        frame = frame.cpu()
                    tracker, detections_class = self.mot_tracking.update(imread[frame_id], frame)
                    frame = []
                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue

                        bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
                        id_num = int(track.track_id)  # Get the ID for the particular track.

                        # Draw bbox from tracker.
                        frame.append(np.concatenate(([id_num + 1], bbox,
                                                     [track.conf_score, track.class_score, track.cid])).reshape(1, -1))
                    if len(frame) > 0:
                        frame = np.concatenate(frame)
                        if self.cuda:
                            frame = torch.from_numpy(frame).cuda()
                    else:
                        frame = torch.empty((0, 8))

                if np.size(frame, 0) == 0:
                    image_handler = ImageHandler(nr_frame, batch_nr, f"frame{nr_frame}",
                                                 imread[frame_id], self.tracking)
                    self.Q_processed.put(image_handler)
                    continue

                image_handler = ImageHandler(nr_frame, batch_nr, f"frame{nr_frame}",
                                             imread[frame_id], self.tracking)
                if self.is_classifier:
                    image_handler.write(frame, self.classifier_classes)
                else:
                    image_handler.write(frame, self.classes)
                self.Q_processed.put(image_handler)

    def get_image(self):
        return self.Q_processed.get()

    def has_images(self):
        return not self.Q_processed.empty()

    def apply_classifier_model(self, imread, frame):
        # get crops from detections in frame
        crops = torch.empty((0, 0))
        detections = frame[:, 1:5]
        for d in detections:
            for i in range(len(d)):
                if d[i] < 0:
                    d[i] = 0
            img_h, img_w, img_ch = imread.shape
            xmin, ymin, xmax, ymax = d
            if xmin > img_w:
                xmin = img_w
            if ymin > img_h:
                ymin = img_h
            ymin = abs(int(ymin))
            ymax = abs(int(ymax))
            xmin = abs(int(xmin))
            xmax = abs(int(xmax))
            try:
                crop = imread[ymin:ymax, xmin:xmax, :]
                crop = prep_image(crop, self.classifier_inp_dim)
                if np.size(crops, 0) == 0:
                    crops = crop
                else:
                    crops = torch.cat((crops, crop))
            except:
                continue
        if self.cuda:
            crops = crops.cuda()
        with torch.no_grad():
            output = self.model_classifier(crops, self.cuda)
        for frame_id in range(np.size(output, 0)):
            new_det = output[frame_id].unsqueeze(0)
            new_det = write_results(new_det, self.classifier_confidence, self.classifier_num_classes,
                                    nms_conf=self.classifier_nms_thesh)
            if np.size(new_det, 0) > 0:
                index = torch.argmax(new_det[:, 6])
                frame[frame_id, 6:8] = new_det[index, 6:8]
            else:
                frame[frame_id, 6] = -1
        frame = frame[frame[:, 6] >= 0]
        return frame


if __name__ == '__main__':
    detect_video()
