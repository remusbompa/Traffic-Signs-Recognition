import torch

import numpy as np
import torchvision

from DeepSort.detection import Detection
from DeepSort.nn_matching import NearestNeighborDistanceMetric
from DeepSort.preprocessing import non_max_suppression
from DeepSort.tracker import Tracker
from scipy.stats import multivariate_normal


def get_gaussian_mask():
    #128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5,0.5])
    sigma = np.array([0.22,0.22])
    covariance = np.diag(sigma**2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z  = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask


class DeepSort:
    def __init__(self, wt_path=None):
        # loading this encoder is slow, should be done only once.
        if wt_path is not None:
            self.encoder = torch.load(wt_path)
        else:
            self.encoder = torch.load('../Siamese/ckpts/model640.pt')

        self.encoder = self.encoder.cuda()
        self.encoder = self.encoder.eval()

        self.metric = NearestNeighborDistanceMetric("cosine", .5, 100)
        self.tracker = Tracker(self.metric)

        self.gaussian_mask = get_gaussian_mask().cuda()

        self.transforms = torchvision.transforms.Compose([ 
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()])

    def reset_tracker(self):
        self.tracker = Tracker(self.metric)

        # Deep sort needs the format `top_left_x, top_left_y, width,height

    def format_yolo_output(self, out_boxes):
        for b in range(len(out_boxes)):
            out_boxes[b][0] = out_boxes[b][0] - out_boxes[b][2] / 2
            out_boxes[b][1] = out_boxes[b][1] - out_boxes[b][3] / 2
        return out_boxes

    def pre_process(self, frame, detections):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()])

        crops = []
        for d in detections:

            for i in range(len(d)):
                if d[i] < 0:
                    d[i] = 0

            img_h, img_w, img_ch = frame.shape

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
                crop = frame[ymin:ymax, xmin:xmax, :]
                crop = transforms(crop)
                crops.append(crop)
            except:
                continue

        crops = torch.stack(crops)

        return crops

    def extract_features_only(self, frame, coords):

        for i in range(len(coords)):
            if coords[i] < 0:
                coords[i] = 0

        img_h, img_w, img_ch = frame.shape

        xmin, ymin, w, h = coords

        if xmin > img_w:
            xmin = img_w

        if ymin > img_h:
            ymin = img_h

        xmax = xmin + w
        ymax = ymin + h

        ymin = abs(int(ymin))
        ymax = abs(int(ymax))
        xmin = abs(int(xmin))
        xmax = abs(int(xmax))

        crop = frame[ymin:ymax, xmin:xmax, :]

        crop = self.transforms(crop)
        crop = crop.cuda()

        gaussian_mask = self.gaussian_mask

        input_ = crop * gaussian_mask
        input_ = torch.unsqueeze(input_, 0)

        features = self.encoder.forward_once(input_)
        features = features.detach().cpu().numpy()

        corrected_crop = [xmin, ymin, xmax, ymax]

        return features, corrected_crop

    def update(self, frame, detections=None):
        if np.size(detections, 0) == 0:
            self.tracker.predict()
            return self.tracker, None

        processed_crops = self.pre_process(frame, detections[:, 1:5]).cuda()
        processed_crops = self.gaussian_mask * processed_crops

        features = self.encoder.forward_once(processed_crops)
        features = features.detach().cpu().numpy()

        if len(features.shape) == 1:
            features = np.expand_dims(features, 0)

        dets = []
        for i in range(len(detections)):
            d = detections[i]
            dets.append(Detection(d[1:5], d[5], features[i], d[7], d[6]))

        outboxes = np.array([d.to_tlwh() for d in dets])

        outscores = np.array([d.confidence for d in dets])
        indices = non_max_suppression(outboxes, 0.8, outscores)

        dets = [dets[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker, dets
