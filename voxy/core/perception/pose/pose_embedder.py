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

from core.structs.attributes import Pose


class KeypointPoseEmbedder:
    """Creates normalized features from 2D keypoint coordinates"""

    def __init__(self, keypoint_names, landmarks, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        self._landmark_names = keypoint_names
        self.landmarks = landmarks

    # This is to be used with AlphaPose
    @classmethod
    def from_pose(cls, pose: Pose):
        keypoint_names = [
            "left_hip",
            "right_hip",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_knee",
            "right_knee",
            "left_wrist",
            "right_wrist",
            "left_ankle",
            "right_ankle",
        ]

        landmarks = np.array(
            [
                np.array([pose.left_hip.x, pose.left_hip.y]),
                np.array([pose.right_hip.x, pose.right_hip.y]),
                np.array([pose.left_shoulder.x, pose.left_shoulder.y]),
                np.array([pose.right_shoulder.x, pose.right_shoulder.y]),
                np.array([pose.left_elbow.x, pose.left_elbow.y]),
                np.array([pose.right_elbow.x, pose.right_elbow.y]),
                np.array([pose.left_knee.x, pose.left_knee.y]),
                np.array([pose.right_knee.x, pose.right_knee.y]),
                np.array([pose.left_wrist.x, pose.left_wrist.y]),
                np.array([pose.right_wrist.x, pose.right_wrist.y]),
                np.array([pose.left_ankle.x, pose.left_ankle.y]),
                np.array([pose.right_ankle.x, pose.right_ankle.y]),
            ]
        )

        return cls(keypoint_names, landmarks)

    def create_features(self):
        assert self.landmarks.shape[0] == len(
            self._landmark_names
        ), f"wrong number of landmarks: {self.landmarks.shape[0]}"

        self._normalize_pose_landmarks()

        embedding = self._keypoint_to_embeddings()

        return embedding

    def create_heuristics(self):

        # TODO: (Nasha) Add these for the bad posture

        self._normalize_pose_landmarks()
        reach_keypoints = {
            "left_shoulder": (self._get_landmark("left_shoulder"))[1],
            "right_shoulder": (self._get_landmark("right_shoulder"))[1],
            "left_elbow": (self._get_landmark("left_elbow"))[1],
            "right_elbow": (self._get_landmark("right_elbow"))[1],
            "left_wrist": (self._get_landmark("left_wrist"))[1],
            "right_wrist": (self._get_landmark("right_wrist"))[1],
        }
        return reach_keypoints

    def _normalize_pose_landmarks(self):
        pose_center = self._get_pose_center()
        self.landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size()
        self.landmarks /= pose_size
        self.landmarks *= 100

    def _get_pose_center(self):
        left_hip = self._get_landmark("left_hip")
        right_hip = self._get_landmark("right_hip")
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self):

        # Hips center.
        hips = self._get_pose_center()

        # Shoulders center.
        left_shoulder = self._get_landmark("left_shoulder")
        right_shoulder = self._get_landmark("right_shoulder")
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center()
        max_dist = np.max(np.linalg.norm(self.landmarks - pose_center, axis=1))

        return max(torso_size * self._torso_size_multiplier, max_dist)

    def _keypoint_to_embeddings(self):
        """Convert pose landmarks into embedding.

        Result:
          Numpy array with pose embedding of shape (M, 2) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array(
            [
                self._diff(
                    self._get_average_by_names("left_hip", "right_hip"),
                    self._get_average_by_names(
                        "left_shoulder", "right_shoulder"
                    ),
                ),
                self._diff_by_names("left_shoulder", "left_elbow"),
                self._diff_by_names("right_shoulder", "right_elbow"),
                self._diff_by_names("left_elbow", "left_wrist"),
                self._diff_by_names("right_elbow", "right_wrist"),
                self._diff_by_names("left_hip", "left_knee"),
                self._diff_by_names("right_hip", "right_knee"),
                self._diff_by_names("left_knee", "left_ankle"),
                self._diff_by_names("right_knee", "right_ankle"),
                self._diff_by_names("left_shoulder", "left_wrist"),
                self._diff_by_names("right_shoulder", "right_wrist"),
                self._diff_by_names("left_hip", "left_ankle"),
                self._diff_by_names("right_hip", "right_ankle"),
                self._diff_by_names("left_hip", "left_wrist"),
                self._diff_by_names("right_hip", "right_wrist"),
                self._diff_by_names("left_shoulder", "left_ankle"),
                self._diff_by_names("right_shoulder", "right_ankle"),
                self._diff_by_names("left_hip", "left_wrist"),
                self._diff_by_names("right_hip", "right_wrist"),
                self._diff_by_names("left_elbow", "right_elbow"),
                self._diff_by_names("left_knee", "right_knee"),
                self._diff_by_names("left_wrist", "right_wrist"),
                self._diff_by_names("left_ankle", "right_ankle"),
                self._diff(
                    self._get_average_by_names("left_wrist", "left_ankle"),
                    self._get_landmark("left_hip"),
                ),
                self._diff(
                    self._get_average_by_names("right_wrist", "right_ankle"),
                    self._get_landmark("right_hip"),
                ),
                (
                    self._get_knee_angles("left"),
                    self._get_knee_angles("right"),
                ),
                (
                    self._get_elbow_angles("left"),
                    self._get_elbow_angles("right"),
                ),
                (
                    self._get_waist_angles("left"),
                    self._get_waist_angles("right"),
                ),
            ]
        )

        return embedding

    def _get_knee_angles(self, left_right):
        v1 = self._diff_by_names(f"{left_right}_knee", f"{left_right}_hip")
        v2 = self._diff_by_names(f"{left_right}_knee", f"{left_right}_ankle")
        cosine_ = np.inner(v1, v2)
        return cosine_ / 100.0

    def _get_elbow_angles(self, left_right):
        v1 = self._diff_by_names(f"{left_right}_elbow", f"{left_right}_wrist")
        v2 = self._diff_by_names(
            f"{left_right}_elbow", f"{left_right}_shoulder"
        )
        cosine_ = np.inner(v1, v2)
        return cosine_ / 100.0

    def _get_waist_angles(self, left_right):
        v1 = self._diff_by_names(f"{left_right}_hip", f"{left_right}_knee")
        v2 = self._diff_by_names(f"{left_right}_hip", f"{left_right}_shoulder")
        cosine_ = np.inner(v1, v2)
        return cosine_ / 100.0

    def _get_landmark(self, landmark_name):
        return self.landmarks[self._landmark_names.index(landmark_name)]

    def _get_average_by_names(self, name_from, name_to):
        lmk_from = self._get_landmark(name_from)
        lmk_to = self._get_landmark(name_to)
        return (lmk_from + lmk_to) * 0.5

    def _diff_by_names(self, name_from, name_to):
        lmk_from = self._get_landmark(name_from)
        lmk_to = self._get_landmark(name_to)
        return self._diff(lmk_from, lmk_to)

    def _diff(self, lmk_from, lmk_to):
        return lmk_to - lmk_from
