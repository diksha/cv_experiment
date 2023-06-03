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


from loguru import logger

from core.perception.pose.pose_embedder import KeypointPoseEmbedder
from core.structs.actor import ActorCategory
from core.structs.body import Body
from core.structs.ergonomics import Activity, ActivityType, PostureType
from core.structs.frame import Frame
from core.utils.math import compare


class LiftClassifier:
    """
    Runs perception lifting algorithms to classify pose as good, bad or random.

    """

    def __init__(
        self,
        classifier_path: str,
        classification_model_type: str = "HM",
    ) -> None:

        if classification_model_type == "DT":
            # TODO : (Nasha) Use onnx to import models
            self._classifier_path = classifier_path
            self._pose_classifier = None
            self._pose_classifier.gpu_id = 0
            self._classifier_type = "HM"
            logger.warning(
                "Decision Tree Models are not supported at this time"
                "using Heuristic Model instead"
            )
        else:
            self._classifier_type = "HM"

        self.BAD_BEND_SPINE_WAIST_ANGLE_THRESHOLD_DEG = 70
        self.LEG_ANGLES_THRESHOLD = 20

    def __call__(self, frame_struct: Frame) -> Frame:
        """
        Executes lift classification alogorithms on each frame

        Args:
            frame_struct(Frame): input frame struct from video

        Returns:
            frame_struct(Frame): output frame struct with actors activity dictionary updated
        """

        for i, actor in enumerate(frame_struct.actors):
            if actor.category == ActorCategory.PERSON:
                body = Body(actor.pose) if actor.pose is not None else None
                if (
                    body is None
                    or not self._is_visible_for_activity_classification(body)
                ):
                    continue
                if frame_struct.actors[i].activity is None:
                    frame_struct.actors[i].activity = {}
                frame_struct.actors[i].activity[
                    ActivityType.LIFTING.name
                ] = self._classify_pose(
                    body,
                    actor.track_id,
                    frame_struct.relative_timestamp_ms,
                )

        return frame_struct

    def _classify_pose(
        self, body: Body, track_id: int, relative_timestamp_ms: int
    ) -> Activity:
        """
        Classify the based on either DT: Decision Tree or HM: Heuristic Model
        Args:

            body(Body): kepypoints with relationships precomputed
            track_id(int): actor track id
            relative_timestamp_ms(int): current frame time struct
        Returns:

            activity(Activity): contains activity type and posture type
        """

        if self._classifier_type == "DT":
            embedder = KeypointPoseEmbedder.from_pose(body.pose)

            features = embedder.create_features().reshape((1, -1))
            lift_type = int(self._pose_classifier.predict(features))
            activity_type = (
                ActivityType.LIFTING
                if lift_type in (PostureType.BAD.value, PostureType.GOOD.value)
                else ActivityType.UNKNOWN
            )
        elif self._classifier_type == "HM":
            lift_type = self._is_lifting_heuristic(
                body, track_id, relative_timestamp_ms
            )

            activity_type = (
                ActivityType.UNKNOWN
                if lift_type == PostureType.UNKNOWN.value
                else ActivityType.LIFTING
            )
        activity = Activity(activity_type, PostureType(lift_type))
        return activity

    def _is_lifting_heuristic(
        self, body: Body, track_id: int, relative_timestamp_ms: int
    ) -> int:
        """
        Lift classification heuristic

        Args:
            body(Body): kepypoints with relationships precomputed
            track_id(int): actor track id
            relative_timestamp_ms(int): current frame time struct

        Returns:
            lift_type(int): Returns one of three lift classes
                0: Bad Lift
                1: Good Lift
                2: Random Pose

        """

        logger.debug(f"Actor {track_id} passed good pose check")
        angle_between_spine_and_legs_deg = None
        if body.is_fully_visible_from_left():
            angle_between_spine_and_legs_deg = body.angle_at_joint(
                body.pose.left_knee,
                body.pose.left_hip,
                body.pose.mid_hip,
                body.pose.neck,
            )
        else:
            angle_between_spine_and_legs_deg = body.angle_at_joint(
                body.pose.right_knee,
                body.pose.right_hip,
                body.pose.mid_hip,
                body.pose.neck,
            )
        # TODO(Nasha): Add checks for occluded actors etc
        are_legs_straight = False
        if body.is_fully_visible_from_left():
            are_legs_straight = compare.is_close(
                body.leftthigh_calves_angle_deg,
                0,
                self.LEG_ANGLES_THRESHOLD,
            )
        elif body.is_fully_visible_from_right():
            are_legs_straight = compare.is_close(
                body.rightthigh_calves_angle_deg,
                0,
                self.LEG_ANGLES_THRESHOLD,
            )
        logger.debug(
            f"At {relative_timestamp_ms} actor {track_id} has:"
            f"angle between spine and legs as {angle_between_spine_and_legs_deg}"
            f"and the legs straight check is {are_legs_straight}"
        )
        # This is a bad lift
        if (
            angle_between_spine_and_legs_deg is not None
            and abs(angle_between_spine_and_legs_deg)
            > self.BAD_BEND_SPINE_WAIST_ANGLE_THRESHOLD_DEG
            and are_legs_straight
        ):
            logger.debug(
                f"Bad Posture Incident detected at {relative_timestamp_ms} !!! "
                f"Offending actor id is {track_id}. "
                f"Angle between spine and legs is {angle_between_spine_and_legs_deg}"
            )
            lift_type = PostureType.BAD.value

        else:
            logger.debug("Random pose detected")

            lift_type = PostureType.UNKNOWN.value

        return lift_type

    def _is_pose_side(self, body: Body) -> bool:
        """
        Checks whether the actor is only visible from the side

        Args:
            body(Body): kepypoints with relationships precomputed
        Returns:
            (bool): True/False check result
        """

        return (
            body.is_fully_visible_from_left()
            != body.is_fully_visible_from_right()
        )

    def _is_visible_for_activity_classification(self, body: Body) -> bool:
        """
        Checks whether the actor is sufficiently visible from either the left or the right side

        Args:
            body(Body): kepypoints with relationships precomputed

        Returns:
            (bool): True/False check for visibilty

        """
        return (
            body.is_fully_visible_from_left()
            or body.is_fully_visible_from_right()
        )
