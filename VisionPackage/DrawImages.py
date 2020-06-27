import os
from random import random
import cv2
import random
import colorsys


def generate_random_bright_color():
    h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
    r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
    return b, g, r


class ImagesHandler:
    def __init__(self, classes, results, imread, det, paths, batch_size):
        self.classes = classes
        # results => [img_ind, left, bottom, right, top, obj_score, class_score, class_ind]
        # the Oy axis is going down (bottom is up and top is down)
        self.results = results.cpu().numpy()
        # read images
        self.imread = imread
        # save destination folder
        self.det = det
        # images paths
        self.paths = paths
        self.batch_size = batch_size
        self.imageList = []
        self.init_images()

    def init_images(self):
        for ind in range(len(self.imread)):
            batch_no = 0
            if ind % self.batch_size:
                batch_no = 1
            batch_no += ind // self.batch_size
            self.imageList.append(ImageHandler(ind, batch_no, self.paths[ind], self.imread[ind]))

    def write(self):
        dict_colors = {}
        for res in self.results:
            c1 = (int(res[1]), int(res[2]))
            c2 = (int(res[3]), int(res[4]))
            cls = int(res[-1])
            img_ind = int(res[0])
            # same color for same classes
            if cls in dict_colors:
                color = dict_colors[cls]
            else:
                color = generate_random_bright_color()
                dict_colors[cls] = color
                self.imageList[img_ind].add_color(cls, color)

            label = f"{self.classes[cls]}"

            image = self.imageList[img_ind]
            cv2.rectangle(image.imread, c1, c2, color, 2)
            cv2.rectangle(image.write_image, c1, c2, color, 1)
            image.add_contour(c1, c2, res[5], res[6], res[7], label, color)
            # the image wrote on the file will be different than the one in ram

            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            # drw rectangle above bottom right corner
            t1 = (c1[0], int(c1[1] - 1.5 * t_size[1]))
            t2 = (int(c1[0] + 1.5 * t_size[0]), c1[1])

            cv2.rectangle(image.write_image, t1, t2, color, -1)
            cv2.putText(image.write_image, label, (c1[0], int(c1[1] - 0.5 * t_size[1])),
                        cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

            t_size = cv2.getTextSize(str(image.crt_no-1), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            t1 = (c1[0], int(c1[1] - 1.5 * t_size[1]))
            t2 = (int(c1[0] + 1.5 * t_size[0]), c1[1])
            cv2.rectangle(image.imread, t1, t2, color, -1)
            cv2.putText(image.imread, str(image.crt_no-1), (c1[0], int(c1[1] - 0.5 * t_size[1])),
                        cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

        for image in self.imageList:
            cv2.imwrite(f"{self.det}/{image.img_name}", image.write_image)


class ImageHandler:
    dict_colors = {}

    def __init__(self, image_ind, batch_no, path, imread, tracking=False):
        self.contours = []
        self.image_ind = image_ind
        self.batch_no = batch_no
        self.path = path
        self.imread = imread
        self.contours = []
        self.colors = {}
        self.write_image = imread.copy()
        self.img_name = os.path.basename(self.path)
        self.batch_info = None
        self.crt_no = 1
        self.shape = (imread.shape[1], imread.shape[0])
        self.tracking = tracking

    def add_contour(self, corner_bl, corner_tr, obj_conf, cls_score, cls, label, color, id_track=0):
        self.contours.append(ContourHandler(self.crt_no, corner_bl, corner_tr, obj_conf, cls_score, cls, label, color,
                                            id_track))
        self.crt_no += 1

    def add_color(self, cls, color):
        self.colors[cls] = color

    def set_batch_info(self, info):
        self.batch_info = info

    def write(self, results, classes):
        for res in results:
            c0 = int(res[0])
            c1 = (int(res[1]), int(res[2]))
            c2 = (int(res[3]), int(res[4]))
            cls = int(res[-1])

            # same color for same classes
            if cls in ImageHandler.dict_colors:
                color = ImageHandler.dict_colors[cls]
            else:
                color = generate_random_bright_color()
                ImageHandler.dict_colors[cls] = color
                self.add_color(cls, color)

            label = f"{classes[cls]}"

            cv2.rectangle(self.imread, c1, c2, color, 2)
            self.add_contour(c1, c2, res[5], res[6], res[7], label, color, c0)
            # the image wrote on the file will be different than the one in ram

            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            # drw rectangle above bottom right corner
            t1 = (c1[0], int(c1[1] - 1.5 * t_size[1]))
            t2 = (int(c1[0] + 1.5 * t_size[0]), c1[1])

            cv2.rectangle(self.imread, t1, t2, color, -1)
            if self.tracking:
                cv2.putText(self.imread, label + f"_{c0}", (c1[0], int(c1[1] - 0.5 * t_size[1])),
                            cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
            else:
                cv2.putText(self.imread, label, (c1[0], int(c1[1] - 0.5 * t_size[1])),
                            cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    def get_last_contour(self):
        if len(self.contours) == 0:
            return None
        return self.contours[-1]


class ContourHandler:
    def __init__(self, crt_no, corner_bl, corner_tr, obj_conf, cls_score, cls, label, color, id_track=0):
        self.number = crt_no
        self.corner_bl = corner_bl
        self.corner_tr = corner_tr
        self.obj_conf = obj_conf
        self.cls_score = cls_score
        self.cls = cls
        self.label = label
        self.color = color
        self.id_track = id_track
