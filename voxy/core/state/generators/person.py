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
from core.structs.ergonomics import ActivityType, PostureType
from core.structs.generator_response import GeneratorResponse
from core.structs.state import State


class PersonStateGenerator(BaseStateGenerator):
    no_ped_zones = []

    def __init__(self, config: dict) -> None:
        self._camera_uuid = config["camera_uuid"]
        self._config = config

    # Helper functions
    def _add_no_ped_zones(self, vignette):
        for actor in vignette.present_frame_struct.actors:
            if actor.category == ActorCategory.NO_PED_ZONE:
                self.no_ped_zones.append(actor)

    def _pedestrian_in_no_ped_zone(self, pedestrian):
        if pedestrian is None:
            return None

        MIN_KEYPOINT_CONFIDENCE = 0.3

        for no_ped_zone in self.no_ped_zones:
            left_ankle_point = (
                pedestrian.pose.left_ankle.to_shapely_point()
                if pedestrian.pose.left_ankle
                and pedestrian.pose.left_ankle.confidence
                >= MIN_KEYPOINT_CONFIDENCE
                else None
            )
            right_ankle_point = (
                pedestrian.pose.right_ankle.to_shapely_point()
                if pedestrian.pose.right_ankle
                and pedestrian.pose.right_ankle.confidence
                >= MIN_KEYPOINT_CONFIDENCE
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

    def _get_states(self, vignette) -> typing.List:
        states = []
        present_relative_timestamp_ms = (
            vignette.present_frame_struct.relative_timestamp_ms
        )

        for tid, tracklet in vignette.tracklets.items():

            # Get Actor from tracklet
            actor = tracklet.get_actor_at_timestamp(
                vignette.present_timestamp_ms
            )
            if actor is None:
                continue

            # Check if actor is person
            if actor is not None and tracklet.category == ActorCategory.PERSON:
                state = State(
                    actor_id=str(tid),
                    track_uuid=actor.track_uuid,
                    actor_category=ActorCategory.PERSON,
                    timestamp_ms=present_relative_timestamp_ms,
                    camera_uuid=self._camera_uuid,
                    person_is_wearing_safety_vest=tracklet.is_believed_to_be_wearing_safety_vest,
                    person_is_wearing_hard_hat=tracklet.is_believed_to_be_wearing_hard_hat,
                    person_is_associated=tracklet.is_associated_with_pit,
                    person_in_no_ped_zone=self._pedestrian_in_no_ped_zone(
                        actor
                    ),
                    person_is_carrying_object=tracklet.get_actor_at_timestamp(
                        present_relative_timestamp_ms
                    ).is_carrying_object,
                    end_timestamp_ms=self._get_end_timestamp_ms(vignette),
                )

                if actor.activity is not None:
                    if ActivityType.LIFTING.name in actor.activity:
                        state.person_lift_type = actor.activity[
                            ActivityType.LIFTING.name
                        ].posture

                    bad_lift_converter = {
                        True: PostureType.BAD,
                        False: PostureType.GOOD,
                        None: PostureType.UNKNOWN,
                    }
                    if ActivityType.REACHING.name in actor.activity:
                        state.person_reach_type = bad_lift_converter[
                            tracklet.is_believed_to_be_in_unsafe_posture
                        ]

                states.append(state)
        return states

    def _get_events(self, vignette):
        events = []
        return events

    def process_vignette(self, vignette) -> GeneratorResponse:
        # Add No Ped Zones from Vignette
        if not self.no_ped_zones:
            self._add_no_ped_zones(vignette)

        # Process States & Events
        states = self._get_states(vignette)
        events = self._get_events(vignette)
        return GeneratorResponse(events=events, states=states)
