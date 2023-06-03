#
# Copyright 2021-2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import typing
import uuid

import torch

from core.structs.actor import Actor, ActorCategory, ActorFactory
from third_party.byte_track.byte_tracker import BYTETracker
from third_party.byte_track.kalman_filter import KalmanFilterConfig


class ByteTrackerConfig:
    """
    Basic byte tracker config to initialize the byte track instance
    """

    mot20 = False
    match_thresh = 0.8
    track_thresh = 0.5
    track_buffer = 30


SEGMENT_2_ACTOR = {
    "OBSTRUCTION": ActorCategory.OBSTRUCTION,
    "SPILL": ActorCategory.SPILL,
}


class SegmenterTracker:
    def __init__(self, camera_uuid: str, category: str, min_pixel_size: int):
        # initialize the tracker when that is available
        self.last_detected_actors = []
        self.tracker = BYTETracker(ByteTrackerConfig(), KalmanFilterConfig())
        self.input_shape = None
        self.camera_uuid = camera_uuid
        self.run_seed = str(uuid.uuid4())
        self.category = category
        self.min_pixel_size = min_pixel_size

    def track(
        self,
        use_previous_actor: bool,
        detection_tensor: torch.Tensor,
        segmentation_mask_shape: tuple,
    ) -> typing.List[Actor]:
        """Track segment of interest

        Args:
            use_previous_actor (bool): use previous actor when we do not run the segmentation
            detection_tensor (torch.Tensor): bboxes around segments (connected componets instances)
            segmentation_mask_shape (tuple): the shape of the segmentation map

        Returns:
            typing.List[Actor]: list of actors
        """
        if use_previous_actor:
            return self.last_detected_actors
        if self.input_shape is None:
            self.input_shape = segmentation_mask_shape
        online_targets = self.tracker.update(
            detection_tensor, self.input_shape, self.input_shape
        )

        actors = [
            ActorFactory.from_detection(
                self.camera_uuid,
                target.track_id,
                target.tlwh,
                SEGMENT_2_ACTOR[self.category],
                target.score,
                run_seed=self.run_seed,
            )
            for target in online_targets
        ]
        self.last_detected_actors = actors
        return actors
