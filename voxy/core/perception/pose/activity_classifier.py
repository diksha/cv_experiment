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

import numpy as np
import torch
from torch import nn

from core.structs.actor import ActorCategory
from core.structs.body import Body
from core.structs.ergonomics import Activity, ActivityType, PostureType


class ActivityModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x


class ActivityClassifier:
    def __init__(
        self,
        classifier_path=None,
    ):
        if classifier_path is not None:

            self._activity_classifier = ActivityModel(30, 5)
            self._activity_classifier.load_state_dict(
                torch.load(classifier_path)
            )
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._keypoints_reference = [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ]

            self._activity_type_dict = {
                0: (ActivityType.LIFTING, PostureType.BAD.value),
                1: (ActivityType.LIFTING, PostureType.GOOD.value),
                2: (ActivityType.REACHING, PostureType.BAD.value),
                3: (ActivityType.REACHING, PostureType.GOOD.value),
                4: (ActivityType.UNKNOWN, PostureType.UNKNOWN.value),
            }
            self._activity_classifier.eval().to(self._device)

    def __call__(self, frame_struct):

        for i, actor in enumerate(frame_struct.actors):
            if actor.category == ActorCategory.PERSON:
                if not actor.pose or not self._is_pose_side(actor.pose):
                    continue
                if frame_struct.actors[i].activity is None:
                    frame_struct.actors[i].activity = {}
                actor_activity = self._classify_activity(actor.pose)
                frame_struct.actors[i].activity[
                    actor_activity.activity.name
                ] = actor_activity
        return frame_struct

    def _classify_activity(self, pose):
        data_p = self._keypoint_tomodel_input(pose)
        data_p = data_p.to(self._device)
        outputs = self._activity_classifier(data_p.float())
        _, preds = torch.max(outputs, 1)
        activity_type, pose_type = self._activity_type_dict[
            preds.data.cpu().item()
        ]
        activity = Activity(activity_type, PostureType(pose_type))
        return activity

    def _get_pose_size(self, keypoints, ratio):
        hips_center = (keypoints[9, :] + keypoints[10, :]) / 2
        shoulders_center = (keypoints[3, :] + keypoints[4, :]) / 2
        torso_size = np.linalg.norm((shoulders_center - hips_center))
        distance = np.linalg.norm((keypoints - hips_center), axis=1)
        max_d = np.max(distance)
        pose_size = max(torso_size * ratio, max_d)
        return pose_size

    def _normalize_pose(self, keypoints):
        data_p = np.expand_dims(np.array(keypoints), axis=1).reshape(-1, 2)
        data_p = np.delete(data_p, [3, 4], axis=0)
        hip_center = (data_p[9, :] + data_p[10, :]) / 2
        data_p = data_p - hip_center
        pose_size = self._get_pose_size(data_p, 2)
        data_p = data_p / pose_size
        return data_p.flatten()

    def _keypoint_tomodel_input(self, pose):
        model_keyp_dict = pose.to_dict()
        keypoints_list = []
        for keyp in self._keypoints_reference:
            keypoints_list.append(model_keyp_dict[keyp]["x"])
            keypoints_list.append(model_keyp_dict[keyp]["y"])
        data_p = torch.unsqueeze(
            torch.tensor(self._normalize_pose(keypoints_list)), 0
        )
        return data_p

    def _is_pose_side(self, pose):
        body = Body(pose)
        return (
            body.is_fully_visible_from_left()
            != body.is_fully_visible_from_right()
        )
