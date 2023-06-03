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

from core.perception.acausal.algorithms.base import BaseAcausalAlgorithm
from core.structs.actor import ActorCategory
from core.structs.body import Body
from core.structs.tracklet import Tracklet


class PPESmootheningAlgorithm(BaseAcausalAlgorithm):
    """Person Pose Estimation uses tracklet information to check whether
    the person is reasonably visible, is a high confidence actor, is pedrestrian
    torso visible and updates the beliefs based on this knowledge.

    Args:
        BaseAcausalAlgorithm (_type_): Base class with process_vignette
    """

    MIN_ACTOR_CONFIDENCE = 0.8

    def __init__(self, config):
        self._config = config
        self.camera_uuid = config["camera_uuid"]
        self.pose_points_min_confidence_threshold = (
            config["perception"]
            .get("acausal_layer", {})
            .get("PPESmoothener", {})
            .get("pose_points_min_confidence_threshold", 0.5)
        )
        self.hard_hat_pred_pct_frames = (
            config["perception"]
            .get("acausal_layer", {})
            .get("PPESmoothener", {})
            .get("hard_hat_pred_pct_frames", 0.4)
        )
        self.hard_hat_time_interval_ms = (
            config["perception"]
            .get("acausal_layer", {})
            .get("PPESmoothener", {})
            .get("hard_hat_time_interval_ms", 2000)
        )
        self.safety_vest_time_interval_ms = (
            config["perception"]
            .get("acausal_layer", {})
            .get("PPESmoothener", {})
            .get("safety_vest_time_interval_ms", 5000)
        )

    def _pedestrian_reasonably_visible(self, tracklet, present_timestamp_ms):
        """Use pose model to determine if the actor is occluded. This function
        returns true if both the actor has confidence > min_confidence
        and pose confidence > pose_points_min_confidence_threshold.

        Returns:
            _type_: True if > body_confidence and > pose_confidence else False
        """
        torso_body_parts = {
            "right_shoulder",
            "left_shoulder",
            "left_hip",
            "right_hip",
        }
        actor_instances = tracklet.get_actor_instances_in_time_interval(
            present_timestamp_ms, present_timestamp_ms
        )
        for actor_instance in list(actor_instances):
            if actor_instance is not None:
                body = Body(actor_instance.pose)
                return (
                    body.is_facing_camera()
                    and body.confidence_above_threshold(
                        torso_body_parts,
                        self.pose_points_min_confidence_threshold,
                    )
                )
        return False

    def _is_high_confidence_actor(
        self, tracklet: Tracklet, present_timestamp_ms: int
    ) -> bool:
        """_is_high_confidence_actor.

        filters the actor based on a threshold of if they are high or low confidence.

        Args:
            tracklet (Tracklet): tracklet for a particular actor
            present_timestamp_ms (int): the current timestamp to query

        Returns:
            bool: whether or not the actor is low confidence
        """
        actor = tracklet.get_actor_at_timestamp(present_timestamp_ms)
        return actor.confidence > self.MIN_ACTOR_CONFIDENCE

    def _pedestrian_torso_visible(self, tracklet, present_timestamp_ms):
        """If the upper body parts are visible with high confidence,
        we assume vest must be visible.

        Args:
            tracklet (_type_): Tracklet helps to track the same actor over multiple
            frames. We need this to check for unique actors in the scene.
            present_timestamp_ms (_type_): time_stamp of the current frame

        Returns:
            _type_: True if kp confidence > min_kp_threshold else False
        """
        upper_body_parts = {
            "right_shoulder",
            "right_elbow",
            "right_wrist",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "left_hip",
            "right_hip",
        }
        sum_kp_confidences = 0.0

        actor = tracklet.get_actor_at_timestamp(present_timestamp_ms)

        if actor is not None:
            for part in upper_body_parts:
                attr = getattr(actor.pose, part)
                if (
                    attr is not None
                    and attr.confidence
                    > self.pose_points_min_confidence_threshold
                ):
                    sum_kp_confidences += attr.confidence

        mean_kp_confidence = float(sum_kp_confidences) / len(upper_body_parts)

        return mean_kp_confidence > self.pose_points_min_confidence_threshold

    def _pedestrian_head_visible(self, tracklet, present_timestamp_ms):
        """If the head parts are visible with high confidence,
        we assume hard hat must be visible.

        Args:
            tracklet (_type_): Tracklet helps to track the same actor over multiple
            frames. We need this to check for unique actors in the scene.
            present_timestamp_ms (_type_): time_stamp of the current frame

        Returns:
            _type_: True if kp confidence > min_kp_threshold else False
        """
        head_parts = {
            "nose",
            "neck",
            "right_eye",
            "left_eye",
            "left_ear",
            "right_ear",
        }
        sum_kp_confidences = 0.0

        actor = tracklet.get_actor_at_timestamp(present_timestamp_ms)

        if actor is not None:
            for part in head_parts:
                attr = getattr(actor.pose, part)
                if (
                    attr is not None
                    and attr.confidence
                    > self.pose_points_min_confidence_threshold
                ):
                    sum_kp_confidences += attr.confidence

        mean_kp_confidence = float(sum_kp_confidences) / len(head_parts)

        return mean_kp_confidence > self.pose_points_min_confidence_threshold

    def _update_safety_vest_belief(self, tracklet, present_timestamp_ms):
        if (
            tracklet.category is not ActorCategory.PERSON
            or present_timestamp_ms is None
        ):
            return

        if not self._pedestrian_reasonably_visible(
            tracklet, present_timestamp_ms
        ):
            return

        # Fetch actors instances self.time_interval ms into the past and self.time_interval ms
        # into the future.
        # If actor "was" not wearing safety vest for that duration, we can reasonably assume
        # the actor is not really wearing safety vest.
        actor_instances = list(
            tracklet.get_actor_instances_in_time_interval(
                present_timestamp_ms - self.safety_vest_time_interval_ms,
                present_timestamp_ms + self.safety_vest_time_interval_ms,
            )
        )
        model_predictions = [
            actor.is_wearing_safety_vest
            for actor in actor_instances
            if actor is not None and actor.is_wearing_safety_vest is not None
        ]
        if len(model_predictions):
            num_predicted_vest = sum(model_predictions)
            tracklet.is_believed_to_be_wearing_safety_vest = (
                num_predicted_vest >= len(model_predictions) // 2
            )
        else:
            tracklet.is_believed_to_be_wearing_safety_vest = None

    def _update_hard_hat_belief(self, tracklet, present_timestamp_ms):
        if (
            tracklet.category is not ActorCategory.PERSON
            or present_timestamp_ms is None
        ):
            return

        if not self._pedestrian_head_visible(tracklet, present_timestamp_ms):
            return

        if not self._is_high_confidence_actor(tracklet, present_timestamp_ms):
            return

        # Fetch actors instances self.time_interval ms into the past and self.time_interval ms
        # into the future.
        # If actor "was" not wearing hard hat for that duration, we can reasonably assume
        # the actor is not really wearing hard hat.
        actor_instances = list(
            tracklet.get_actor_instances_in_time_interval(
                present_timestamp_ms - self.hard_hat_time_interval_ms,
                present_timestamp_ms + self.hard_hat_time_interval_ms,
            )
        )

        wearing_hat_values = [
            actor.is_wearing_hard_hat
            for actor in actor_instances
            if actor is not None and actor.is_wearing_hard_hat is not None
        ]

        if len(wearing_hat_values):
            num_predicted_hat = sum(wearing_hat_values)
            tracklet.is_believed_to_be_wearing_hard_hat = (
                num_predicted_hat
                >= len(wearing_hat_values) * self.hard_hat_pred_pct_frames
            )
        else:
            tracklet.is_believed_to_be_wearing_hard_hat = None

    def process_vignette(self, vignette):
        """Update beliefs for hard_hat and safety_vest for the tracklet

        Args:
            vignette (_type_): Object vignette contains all the scene information

        Returns:
            _type_: Vignette
        """
        for _, tracklet in vignette.tracklets.items():
            self._update_safety_vest_belief(
                tracklet, vignette.present_timestamp_ms
            )
            self._update_hard_hat_belief(
                tracklet, vignette.present_timestamp_ms
            )
        return vignette
