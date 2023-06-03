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
"""
"""

# For python 3 compatibility
from __future__ import absolute_import, division, print_function

import math

import numpy as np

from core.structs.attributes import KeyPoint
from core.structs.line_segment import LineSegment, Point
from core.utils.constants import (
    CONFIDENCE_SCORE_THRESHOLD,
    FILTER_THRESHOLD_FOR_LOW_CONFIDENCE_KEYPOINTS,
    FULL_VIEW_MODE_ON,
)
from core.utils.math import compare


class Keypoint:
    """ """

    def __init__(self, keypoint):
        self.x = int(keypoint[0])
        self.y = int(keypoint[1])
        self.c = keypoint[2]

    def __str__(self):
        return "X: {}, Y: {}, C: {}".format(self.x, self.y, self.c)


def convert_keypoint_to_point(kp1):
    return Point(kp1.x, kp1.y)


lower_body_parts = {
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "right_big_toe",
}


def are_lower_body_parts_visible(body):
    number_low_confidence_keypoints = 0
    for part in lower_body_parts:
        attr = getattr(body, part)
        if attr.c < CONFIDENCE_SCORE_THRESHOLD:
            number_low_confidence_keypoints = (
                number_low_confidence_keypoints + 1
            )

    if (
        number_low_confidence_keypoints
        >= FILTER_THRESHOLD_FOR_LOW_CONFIDENCE_KEYPOINTS
    ):
        return False

    return True


class Body:
    def __init__(self, pose):
        self.pose = pose
        # compute neck and mid hip from left and right shoulder and hip respectively.
        self.pose.neck = KeyPoint(
            x=(self.pose.left_shoulder.x + self.pose.right_shoulder.x) / 2,
            y=(self.pose.left_shoulder.y + self.pose.right_shoulder.y) / 2,
            confidence=(
                self.pose.left_shoulder.confidence
                + self.pose.right_shoulder.confidence
            )
            / 2,
        )
        self.pose.mid_hip = KeyPoint(
            x=(self.pose.left_hip.x + self.pose.right_hip.x) / 2,
            y=(self.pose.left_hip.y + self.pose.right_hip.y) / 2,
            confidence=(
                self.pose.left_hip.confidence + self.pose.right_hip.confidence
            )
            / 2,
        )

        self._precompute_angles_at_joints()

    def _precompute_angles_at_joints(self):
        self.spine_leftbicep_angle_deg = abs(
            self.angle_at_joint(
                self.pose.neck,
                self.pose.mid_hip,
                self.pose.left_shoulder,
                self.pose.left_elbow,
            )
        )
        self.spine_rightbicep_angle_deg = abs(
            self.angle_at_joint(
                self.pose.neck,
                self.pose.mid_hip,
                self.pose.right_shoulder,
                self.pose.right_elbow,
            )
        )
        self.leftbicep_forearm_angle_deg = abs(
            self.angle_at_joint(
                self.pose.left_shoulder,
                self.pose.left_elbow,
                self.pose.left_elbow,
                self.pose.left_wrist,
            )
        )
        self.rightbicep_forearm_angle_deg = abs(
            self.angle_at_joint(
                self.pose.right_shoulder,
                self.pose.right_elbow,
                self.pose.right_elbow,
                self.pose.right_wrist,
            )
        )
        self.leftthigh_calves_angle_deg = abs(
            self.angle_at_joint(
                self.pose.left_hip,
                self.pose.left_knee,
                self.pose.left_knee,
                self.pose.left_ankle,
            )
        )
        self.rightthigh_calves_angle_deg = abs(
            self.angle_at_joint(
                self.pose.right_hip,
                self.pose.right_knee,
                self.pose.right_knee,
                self.pose.right_ankle,
            )
        )
        self.spine_hip_angle_deg = abs(
            self.angle_at_joint(
                self.pose.neck,
                self.pose.mid_hip,
                self.pose.left_hip,
                self.pose.right_hip,
            )
        )

    def _generate_vectors_from_keypoints(self, kp1, kp2, kp3, kp4):
        v1_x = kp2.x - kp1.x
        v1_y = kp2.y - kp1.y
        v2_x = kp4.x - kp3.x
        v2_y = kp4.y - kp3.y
        v1 = [v1_x, v1_y]
        v2 = [v2_x, v2_y]
        return v1, v2

    """
    Returns the angle between the vector formed by kp1 and kp2 and the vector
    formed by kp3 and kp4. The resultant angle is returned in degrees with a range
    of [-180, 180]. The angle is positive for clockwise  movement from vector
    formed by kp1 and kp2 and negative for anti-clockwise.
    """

    def angle_at_joint(self, kp1, kp2, kp3, kp4):
        v0, v1 = self._generate_vectors_from_keypoints(kp1, kp2, kp3, kp4)
        angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        return math.degrees(angle)

    def are_arms_close_to_body(self, error_threshold):
        return compare.is_close(
            self.spine_leftbicep_angle_deg, 0, error_threshold
        ) and compare.is_close(
            self.spine_rightbicep_angle_deg, 0, error_threshold
        )

    def are_legs_straight(
        self, error_threshold=0, full_body_enabled=FULL_VIEW_MODE_ON
    ):
        # If we can't see the legs assume they are straight
        if not full_body_enabled:
            return True
        return compare.is_close(
            self.leftthigh_calves_angle_deg, 0, error_threshold
        ) and compare.is_close(
            self.rightthigh_calves_angle_deg, 0, error_threshold
        )

    def are_arms_straight(self, error_threshold=0):
        return compare.is_close(
            self.leftbicep_forearm_angle_deg, 0, error_threshold
        ) and compare.is_close(
            self.rightbicep_forearm_angle_deg, 0, error_threshold
        )

    def get_spine(self):
        return LineSegment(
            convert_keypoint_to_point(self.neck),
            convert_keypoint_to_point(self.mid_hip),
        )

    def get_left_shoulder(self):
        return LineSegment(
            convert_keypoint_to_point(self.neck),
            convert_keypoint_to_point(self.left_shoulder),
        )

    def get_right_shoulder(self):
        return LineSegment(
            convert_keypoint_to_point(self.neck),
            convert_keypoint_to_point(self.right_shoulder),
        )

    def get_left_bicep(self):
        return LineSegment(
            convert_keypoint_to_point(self.left_shoulder),
            convert_keypoint_to_point(self.left_elbow),
        )

    def get_right_bicep(self):
        return LineSegment(
            convert_keypoint_to_point(self.right_shoulder),
            convert_keypoint_to_point(self.right_elbow),
        )

    def get_left_forearm(self):
        return LineSegment(
            convert_keypoint_to_point(self.left_elbow),
            convert_keypoint_to_point(self.left_wrist),
        )

    def get_right_forearm(self):
        return LineSegment(
            convert_keypoint_to_point(self.right_elbow),
            convert_keypoint_to_point(self.right_wrist),
        )

    def get_left_thigh(self):
        return LineSegment(
            convert_keypoint_to_point(self.left_hip),
            convert_keypoint_to_point(self.left_knee),
        )

    def get_right_thigh(self):
        return LineSegment(
            convert_keypoint_to_point(self.right_hip),
            convert_keypoint_to_point(self.right_knee),
        )

    def get_left_calf(self):
        return LineSegment(
            convert_keypoint_to_point(self.left_knee),
            convert_keypoint_to_point(self.left_ankle),
        )

    def get_right_calf(self):
        return LineSegment(
            convert_keypoint_to_point(self.right_knee),
            convert_keypoint_to_point(self.right_ankle),
        )

    def get_left_arm(self):
        return (self.get_left_bicep(), self.get_left_forearm())

    def get_right_arm(self):
        return (self.get_right_bicep(), self.get_right_forearm())

    def get_left_leg(self):
        return (self.get_left_thigh(), self.get_left_calf())

    def get_right_leg(self):
        return (self.get_right_thigh(), self.get_right_calf())

    def confidence_above_threshold(self, relevant_keypoints, min_threshold):
        pose_dict = self.pose.to_dict()
        for key, kp in pose_dict.items():
            if key in relevant_keypoints:
                if kp is None or kp["confidence"] < min_threshold:
                    return False
        return True

    def is_fully_visible_from_left(self, min_threshold=0.5):
        relevant_keypoints = [
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "left_hip",
            "left_knee",
            "left_ankle",
            "right_shoulder",
            "right_hip",
        ]
        return self.confidence_above_threshold(
            relevant_keypoints, min_threshold
        )

    def is_fully_visible_from_right(self, min_threshold=0.5):
        relevant_keypoints = [
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            "right_hip",
            "right_knee",
            "right_ankle",
            "left_shoulder",
            "left_hip",
        ]
        return self.confidence_above_threshold(
            relevant_keypoints, min_threshold
        )

    def is_facing_camera(self):
        if (
            self.pose.right_shoulder is not None
            and self.pose.left_shoulder is not None
        ):
            return self.pose.right_shoulder.x < self.pose.left_shoulder.x
        if self.pose.right_hip is not None and self.pose.left_hip is not None:
            return self.pose.right_hip.x < self.pose.left_hip.x
        if (
            self.pose.right_elbow is not None
            and self.pose.left_elbow is not None
        ):
            return self.pose.right_elbow.x < self.pose.left_elbow.x
        if (
            self.pose.right_ankle is not None
            and self.pose.left_ankle is not None
        ):
            return self.pose.right_ankle.x < self.pose.left_ankle.x
        # Give up. Returning False is incorrect behavior. Fix it. TODO(harishma)
        return False
