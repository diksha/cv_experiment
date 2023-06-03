#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import os
from copy import deepcopy

import attr
import numpy as np
import torch

from .impl.deep.feature_extractor import Extractor
from .impl.sort.detection import Detection
from .impl.sort.nn_matching import NearestNeighborDistanceMetric
from .impl.sort.preprocessing import non_max_suppression
from .impl.sort.tracker import Tracker
from core.cv.tracker.base import TrackerBase
from core.cv.tracker.base import TrackerBuilderBase

"""
from core.labeling.loader import Loader
from core.cv.detector.api import YOLODetector
from core.cv.tracker.deep_sort import DeepSort
import cv2

detector = YOLODetector()
vcap = cv2.VideoCapture(Loader().get_video_url('8d10d83c-0b68-4e37-b31b-5e716ec94c30'))
tracker = DeepSort()

for i in range(10):
    ret, frame = vcap.read()
    boxes, scores = detector(frame)
    print(tracker.update(boxes, scores, frame))

"""


class TrackerDeepSort(TrackerBase):
    def __init__(
        self,
        model_path=os.path.join(
            os.environ["BUILD_WORKSPACE_DIRECTORY"], "data/artifacts/ckpt.t7"
        ),
        max_dist=0.2,
        min_confidence=0.3,
        nms_max_overlap=1.0,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100,
        use_cuda=True,
    ):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.extractor = Extractor(model_path, use_cuda=use_cuda)
        max_cosine_distance = max_dist
        nn_budget = nn_budget
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init
        )

    def update(self, boxes, confidences, ori_img, poses):
        bbox_tlwh = deepcopy(boxes)
        # bbox_xywh[:, 2] = bbox_xywh[:, 2] - bbox_xywh[:, 0] + 1
        # bbox_xywh[:, 3] = bbox_xywh[:, 3] - bbox_xywh[:, 1] + 1

        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_tlwh, ori_img)
        # bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], confidences[i], features[i], poses[i])
            for i, conf in enumerate(confidences)
            if conf > self.min_confidence
        ]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        poses = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            x, y, w, h, vx, vy = track.to_tlwh_and_velocity()
            # x1,y1,x2,y2 = self._convert_box([x, y, w, h])
            track_id = track.track_id
            outputs.append(
                np.array(
                    [x, y, w, h, vx, vy, track.confidence, track_id], dtype=np.float32
                )
            )
            poses.append(track.pose)
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs, poses

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _convert_box(self, bbox_tlwh):
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        w1 = (x2 - x1) / 2
        h1 = (y2 - y1) / 2
        return x1 + w1, y1 + h1, x2 + w1, y2 + h1

    def _get_features(self, bbox_tlwh, ori_img):
        im_crops = []
        for box in bbox_tlwh:
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


@attr.s(slots=True)
class TrackerDeepSortBuilder(TrackerBuilderBase):
    def __init__(self):
        super().__init__()

    def __call__(self, **kwargs):
        return TrackerDeepSort()

    def validate_inputs(self, config_file):
        pass
