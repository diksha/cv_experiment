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


class NoPedZoneStateGenerator(BaseStateGenerator):
    """
    Generator for no ped zone states and events
    """

    no_ped_zones = []
    MIN_KEYPOINT_CONFIDENCE = 0.3

    def __init__(self, config: dict):
        self._camera_uuid = config["camera_uuid"]

    # Helper functions
    def _add_no_ped_zones(self, vignette):
        for actor in vignette.present_frame_struct.actors:
            if actor.category == ActorCategory.NO_PED_ZONE:
                self.no_ped_zones.append(actor)

    def _pedestrian_in_no_ped_zone(self, pedestrian, no_ped_zone) -> bool:
        """Returns if the person is in the no ped zone

        Args:
            pedestrian (Actor): Actor of category PERSON
            no_ped_zone (Actor): Actor of category NO_PED_ZONE

        Returns:
            bool: whether the pedestrian is in the no ped zone
        """
        if pedestrian is None:
            return None

        left_ankle_point = (
            pedestrian.pose.left_ankle.to_shapely_point()
            if pedestrian.pose.left_ankle
            and pedestrian.pose.left_ankle.confidence
            >= self.MIN_KEYPOINT_CONFIDENCE
            else None
        )
        right_ankle_point = (
            pedestrian.pose.right_ankle.to_shapely_point()
            if pedestrian.pose.right_ankle
            and pedestrian.pose.right_ankle.confidence
            >= self.MIN_KEYPOINT_CONFIDENCE
            else None
        )

        if (
            left_ankle_point
            and no_ped_zone.polygon.to_shapely_polygon().contains(
                left_ankle_point
            )
            or right_ankle_point
            and no_ped_zone.polygon.to_shapely_polygon().contains(
                right_ankle_point
            )
        ):
            return True

        return False

    def _get_states(self, vignette: Vignette) -> typing.List[State]:
        """
        Get states for no ped zones
        Args:
            vignette (Vignette): vignette with actor tracklets
        Returns:
            typing.List[State]: list of States for no ped zones
        """
        states = []
        for no_ped_zone in self.no_ped_zones:
            num_persons_in_no_ped_zone = 0
            for tracklet in vignette.tracklets.values():
                actor = tracklet.get_actor_at_timestamp(
                    vignette.present_timestamp_ms
                )

                if tracklet.category == ActorCategory.PERSON:
                    if self._pedestrian_in_no_ped_zone(actor, no_ped_zone):
                        num_persons_in_no_ped_zone += 1

            # update state with count
            states.append(
                State(
                    actor_id=str(no_ped_zone.track_id),
                    track_uuid=no_ped_zone.track_uuid,
                    actor_category=ActorCategory.NO_PED_ZONE,
                    timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                    camera_uuid=self._camera_uuid,
                    num_persons_in_no_ped_zone=num_persons_in_no_ped_zone,
                    end_timestamp_ms=self._get_end_timestamp_ms(vignette),
                )
            )

        return states

    def process_vignette(self, vignette: Vignette) -> GeneratorResponse:
        """
        Callable for generator to generate states for no ped zones from the vigneete
        Args:
            vignette (Vignette): vignette with actor tracklets
        Returns:
            GeneratorResponse: generator response struct
        """
        # Add No Ped Zones from Vignette
        if not self.no_ped_zones:
            self._add_no_ped_zones(vignette)

        # Process States & Events
        states = self._get_states(vignette)
        return GeneratorResponse(events=[], states=states)
