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

import uuid
from typing import Any

import numpy as np

from core.structs.actor import ActorCategory, ActorFactory
from third_party.byte_track.byte_tracker import BYTETracker
from third_party.byte_track.kalman_filter import KalmanFilterConfig


class ByteTrackerConfig:
    mot20 = False
    match_thresh = 0.8
    track_thresh = 0.5
    track_buffer = 30


class DetectorTracker:
    """DetectorTracker.

    this is the main tracker algorithm to detect and track actor bounding boxes
    The main approach is based on the byte tracker algorithm. To run the tracker,
    you have to pass in a detector that will output tracks in this dictionary format:

    {
        [person_category] : [detected bounding boxes],
        [pit_category] :    [detected bounding boxes],
        ....
    }

    The tracker will output actor bounding boxes with specific track ids

    """

    def __init__(self, camera_uuid: str, detector: Any) -> None:
        self.detector = detector
        self.input_shape = detector.get_input_shape()
        self.actor_types = detector.get_actor_categories()
        self.camera_uuid = camera_uuid
        self.trackers = {}
        self.run_seed = str(uuid.uuid4())

        kf_configuration = self.get_kalman_filter_configuration()
        # initialize trackers
        for actor_category in self.actor_types:
            self.trackers[actor_category] = BYTETracker(
                ByteTrackerConfig(), kf_configuration[actor_category]
            )

    def get_kalman_filter_configuration(self) -> dict:
        """get_kalman_filter_configuration.

        Gets the kalman filter configuration indexed by the actor type

        Returns:
            dict: the configuration indexed by the actor type
        """
        # TODO make the configuration here a bit more mature
        configuration = {}
        # PIT:
        configuration[ActorCategory.PIT] = KalmanFilterConfig()
        configuration[ActorCategory.PIT].ASPECT_RATIO_STD_NOISE = 5e-2
        configuration[ActorCategory.PIT].ASPECT_RATIO_VELOCITY_STD_NOISE = 5e-5
        # Humans, Hat, Safetyvest (default):
        configuration[ActorCategory.PERSON] = KalmanFilterConfig()
        configuration[ActorCategory.HARD_HAT] = KalmanFilterConfig()
        configuration[ActorCategory.SAFETY_VEST] = KalmanFilterConfig()
        return configuration

    def __call__(self, frame: np.array, frame_epoch_time_ms: int) -> list:
        """
        __call__.

        Generates the current set of actors in the frame

        Args:
            frame (np.array): the current image to detect and track object
            frame_epoch_time_ms (int): the current timestamp of the image (currently not used, but here for compatibility)

        Returns:
            list: a list of actors in the frame
        """
        detections = self.detector.predict(frame)
        # TODO parallelize this maybe down the line
        actors = []
        for actor_category in detections:
            # TODO find out what the
            bounding_box_predictions = detections[actor_category]
            online_targets = self.trackers[actor_category].update(
                bounding_box_predictions, self.input_shape, self.input_shape
            )
            # get all these actors
            new_actors = self.convert_targets_to_actors(
                online_targets, actor_category
            )
            actors.extend(new_actors)
        return actors

    def convert_targets_to_actors(
        self, targets: list, actor_category: ActorCategory
    ) -> list:
        """convert_targets_to_actors.

        this converts the raw list of targets of a particular category into a set of actors

        Args:
            targets (list): this is a list of raw targets from bytetrack
            actor_category (ActorCategory): the particular category of object for the target list

        Returns:
            list: a list of actors
        """
        # create an actor
        # TODO see if we need to redefine these
        actors = []
        for target in targets:
            actor = ActorFactory.from_detection(
                self.camera_uuid,
                target.track_id,
                target.tlwh,
                actor_category,
                target.score,
                run_seed=self.run_seed,
            )
            actors.append(actor)
        return actors
