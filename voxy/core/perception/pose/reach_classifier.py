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

# trunk-ignore(bandit/B403)
import pickle

import numpy as np
import torch
from torch import nn

from core.perception.pose.pose_embedder import KeypointPoseEmbedder
from core.structs.actor import ActorCategory
from core.structs.body import Body
from core.structs.ergonomics import Activity, ActivityType, PostureType
from lib.ml.inference.factory.base import InferenceBackendType
from lib.ml.inference.tasks.ergonomic_overreach.fully_connected_v1.factory import (
    FullyConnectedV1InferenceProviderFactory,
)

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class ReachModel(nn.Module):
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


class ReachClassifier:
    def __init__(
        self,
        classifier_path,
        gpu_runtime: GpuRuntimeBackend,
        triton_server_url: str,
        threshold=2,
        classification_model_type="HM",
    ):
        if classifier_path is not None and classification_model_type == "DT":
            with open(classifier_path, "rb") as f:
                # TODO : (Nasha) Use onnx to import models
                # trunk-ignore(bandit/B301)
                self._pose_classifier = pickle.load(f)
        elif classifier_path is not None and classification_model_type == "DL":
            self.inference_provider = FullyConnectedV1InferenceProviderFactory(
                local_inference_provider_type=InferenceBackendType.TORCHSCRIPT,
                gpu_runtime=gpu_runtime,
                triton_server_url=triton_server_url,
            ).get_inference_provider(classifier_path)
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

            self._reach_type_dict = {
                0: (ActivityType.REACHING, PostureType.BAD.value),
                1: (ActivityType.UNKNOWN, PostureType.UNKNOWN.value),
            }
        self.threshold = threshold
        self._classifier_type = classification_model_type
        self.BAD_POSTURE_SHOULDER_ELBOW_ANGLE_THRESHOLD_DEG = 110
        self.GOOD_POSTURE_SHOULDER_ELBOW_ANGLE_THRESHOLD_DEG = 90

    def __call__(self, frame_struct):

        for i, actor in enumerate(frame_struct.actors):
            if actor.category == ActorCategory.PERSON:
                body = Body(actor.pose)
                if not actor.pose or not self._is_pose_visible(body):
                    continue
                if frame_struct.actors[i].activity is None:
                    frame_struct.actors[i].activity = {}
                frame_struct.actors[i].activity[
                    ActivityType.REACHING.name
                ] = self._classify_pose(actor.pose, actor.track_id)
        return frame_struct

    def _classify_pose(self, pose, track_id):
        embedder = KeypointPoseEmbedder.from_pose(pose)
        if self._classifier_type == "DT":
            features = embedder.create_features().reshape((1, -1))
            # pose_classes are [0: "bad reach", 1: "random pose"]
            reach_type = int(self._pose_classifier.predict(features))
            activity_type = (
                ActivityType.REACHING
                if reach_type == PostureType.BAD.value
                else ActivityType.UNKNOWN
            )
        elif self._classifier_type == "DL":
            data_p = self._keypoint_tomodel_input(pose)
            outputs = self.inference_provider.process(data_p)
            _, preds = torch.max(outputs, 1)
            activity_type, reach_type = self._reach_type_dict[
                preds.data.cpu().item()
            ]
        elif self._classifier_type == "HM":
            # pose_classes are [0: "bad reach", 1: "good reach", 2: "random pose"]
            reach_type = self._is_reaching(pose)

            activity_type = (
                ActivityType.REACHING
                if reach_type
                in (PostureType.BAD.value, PostureType.GOOD.value)
                else ActivityType.UNKNOWN
            )
        else:
            raise ValueError("Model Type is Not Recognized!")
        activity = Activity(activity_type, PostureType(reach_type))
        return activity

    def _is_reaching(self, pose):

        body = Body(pose)
        if self._is_pose_visible(body):
            angle_between_shoulder_and_elbow_deg = None
            if body.is_fully_visible_from_left():
                angle_between_shoulder_and_elbow_deg = body.angle_at_joint(
                    body.pose.left_shoulder,
                    body.pose.left_elbow,
                    body.pose.left_shoulder,
                    body.pose.left_hip,
                )
            elif body.is_fully_visible_from_right():
                angle_between_shoulder_and_elbow_deg = body.angle_at_joint(
                    body.pose.right_shoulder,
                    body.pose.right_elbow,
                    body.pose.right_shoulder,
                    body.pose.right_hip,
                )
            if (
                angle_between_shoulder_and_elbow_deg is not None
                and abs(angle_between_shoulder_and_elbow_deg)
                >= self.BAD_POSTURE_SHOULDER_ELBOW_ANGLE_THRESHOLD_DEG
            ):
                reach_type = 0
            elif (
                angle_between_shoulder_and_elbow_deg is not None
                and abs(angle_between_shoulder_and_elbow_deg)
                < self.BAD_POSTURE_SHOULDER_ELBOW_ANGLE_THRESHOLD_DEG
                and abs(angle_between_shoulder_and_elbow_deg)
                > self.GOOD_POSTURE_SHOULDER_ELBOW_ANGLE_THRESHOLD_DEG
            ):
                reach_type = 1
            else:
                reach_type = 2

        return reach_type

    def _is_pose_side(self, pose):
        body = Body(pose)
        return (
            body.is_fully_visible_from_left()
            != body.is_fully_visible_from_right()
        )

    def _is_pose_visible(self, body: Body) -> bool:
        return (
            body.is_fully_visible_from_left()
            or body.is_fully_visible_from_right()
        )

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
