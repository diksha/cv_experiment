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

import typing

from core.state.generators.base import BaseStateGenerator
from core.structs.actor import ActorCategory
from core.structs.generator_response import GeneratorResponse
from core.structs.state import State
from core.structs.vignette import Vignette


class MotionDetectionZoneStateGenerator(BaseStateGenerator):
    """
    Generator for motion detection zone states and events
    """

    def __init__(self, config: dict):
        self._camera_uuid = config["camera_uuid"]

    def _get_states(self, vignette: Vignette) -> typing.List[State]:
        """
        Get states for motion detection zones
        Args:
            vignette (Vignette): vignette with actor tracklets
        Returns:
            typing.List[State]: list of States for motion detection zones
        """
        states = []
        for tracklet in vignette.tracklets.values():
            if tracklet.category == ActorCategory.MOTION_DETECTION_ZONE:
                states.append(
                    State(
                        actor_id=tracklet.get_actors()[-1].track_uuid,
                        track_uuid=tracklet.get_actors()[-1].track_uuid,
                        actor_category=ActorCategory.MOTION_DETECTION_ZONE,
                        timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                        camera_uuid=self._camera_uuid,
                        motion_zone_is_in_motion=tracklet.is_motion_zone_in_motion,
                        end_timestamp_ms=self._get_end_timestamp_ms(vignette),
                    )
                )
        return states

    def process_vignette(self, vignette: Vignette) -> GeneratorResponse:
        """
        Callable for generator to generate states for motion detection zones from the vigneete
        Args:
            vignette (Vignette): vignette with actor tracklets
        Returns:
            GeneratorResponse: generator response struct
        """
        states = self._get_states(vignette)
        return GeneratorResponse(events=[], states=states)
