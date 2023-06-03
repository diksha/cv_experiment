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
from core.structs.actor import Actor, ActorCategory
from core.structs.event import Event, EventType
from core.structs.generator_response import GeneratorResponse
from core.structs.state import State
from core.structs.tracklet import Tracklet
from core.structs.vignette import Vignette


class PitStateGenerator(BaseStateGenerator):
    DRIVABLE_REGION_OVERLAP_THRESHOLD = 0.5

    def __init__(self, config: dict) -> None:
        self._camera_uuid = config["camera_uuid"]
        # load configs
        self.config = config
        # get default drivable region overlap
        drivable_region_overlap_config = (
            self.config.get("state", {})
            .get("pit", {})
            .get("drivable_region_threshold")
        )
        if drivable_region_overlap_config is not None:
            self.DRIVABLE_REGION_OVERLAP_THRESHOLD = (
                drivable_region_overlap_config
            )

    def is_in_driving_area(self, actor: Actor, drivable_regions: list) -> bool:
        """is_in_driving_area.

        This returns true if the pit is in a driving area

        Args:
            actor (Actor): the query pit
            drivable_regions (list): the list of drivable regions

        Returns:
            bool: if the pit is in the drivable region or not
        """
        if actor is None:
            return None
        shapely_actor = actor.get_shapely_polygon()
        if not drivable_regions:
            # Drivable regions don't apply here, so this should be handled upstream in the incident monitors
            return None

        for area in drivable_regions:
            shapely_area = area.get_shapely_polygon()
            intersect_area = shapely_actor.intersection(shapely_area).area
            # At least 50% of actor is in driving area
            if intersect_area > (
                shapely_actor.area * self.DRIVABLE_REGION_OVERLAP_THRESHOLD
            ):
                return True
        # Location is acceptable
        return False

    def get_drivable_areas(self, vignette: Vignette) -> list:
        """get_drivable_areas.

        returns the list of drivable regions from the list of actors

        Args:
            vignette (Vignette): vignette

        Returns:
            list:
        """
        drivable_regions = []
        for actor in vignette.present_frame_struct.actors:
            if actor.category == ActorCategory.DRIVING_AREA:
                drivable_regions.append(actor)
        return drivable_regions

    def _get_states(self, vignette: Vignette) -> typing.List:

        states = []
        drivable_areas = self.get_drivable_areas(vignette)

        for tid, tracklet in vignette.tracklets.items():
            if tracklet.category == ActorCategory.PIT:
                current_time = (
                    vignette.present_frame_struct.relative_timestamp_ms
                )
                current_actor = tracklet.get_actor_at_timestamp(current_time)
                if current_actor is None:
                    continue
                states.append(
                    State(
                        actor_id=str(tid),
                        track_uuid=current_actor.track_uuid,
                        actor_category=ActorCategory.PIT,
                        timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                        camera_uuid=self._camera_uuid,
                        pit_is_stationary=self._tracklet_stationary(
                            tracklet, vignette
                        ),
                        end_timestamp_ms=self._get_end_timestamp_ms(vignette),
                        pit_is_associated=tracklet.is_associated_with_person,
                        pit_in_driving_area=self.is_in_driving_area(
                            current_actor, drivable_areas
                        ),
                    )
                )

        return states

    def _tracklet_stationary(
        self, tracklet: Tracklet, vignette: Vignette
    ) -> bool:
        return tracklet.is_stationary

    def _get_pit_crossing_events_for_intersection(
        self, intersection: Actor, vignette: Vignette, max_lookback: int = None
    ) -> list:
        """Generate PIT Entering or PIT Exiting Intersection events."""
        events = []
        for (
            track_id,
            tracklet,
        ) in vignette.tracklets.items():
            if tracklet.category != ActorCategory.PIT:
                continue

            pit = tracklet.get_actor_at_timestamp(
                vignette.present_timestamp_ms
            )
            if pit is None:
                continue
            pit_track_id = track_id
            # First check if an actor is intersects an intersection. This indicates the
            # actor is currently on the side of the door away from the camera.
            pit_partially_contained_now = (
                intersection.get_shapely_polygon().intersects(
                    pit.get_shapely_polygon()
                )
            )

            stop_search = False
            # start from the most recent history and move backwards
            for index, historical_frame_struct in enumerate(
                vignette.past_frame_structs[::-1]
            ):
                if (max_lookback and index >= max_lookback) or stop_search:
                    break
                for historical_actor in historical_frame_struct.actors:
                    if historical_actor.track_id == pit_track_id:
                        pit_partially_contained_in_past = (
                            intersection.get_shapely_polygon().intersects(
                                historical_actor.get_shapely_polygon()
                            )
                        )
                        if (
                            pit_partially_contained_in_past
                            != pit_partially_contained_now
                        ):
                            event_type = (
                                EventType.PIT_EXITING_INTERSECTION
                                if pit_partially_contained_now
                                else EventType.PIT_ENTERING_INTERSECTION
                            )
                            events.append(
                                Event(
                                    timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                                    camera_uuid=self._camera_uuid,
                                    subject_id=str(pit.track_id),
                                    subject_uuid=pit.track_uuid,
                                    event_type=event_type,
                                    object_id=str(intersection.track_id),
                                    object_uuid=intersection.track_uuid,
                                    end_timestamp_ms=self._get_end_timestamp_ms(
                                        vignette
                                    ),
                                    x_velocity_pixel_per_sec=tracklet.x_velocity_pixel_per_sec,
                                    y_velocity_pixel_per_sec=tracklet.x_velocity_pixel_per_sec,
                                    normalized_speed=tracklet.normalized_pixel_speed,
                                )
                            )
                        stop_search = True
        return events

    def _get_pit_crossing_events_for_aisle_end(
        self, aisle_end: Actor, vignette: Vignette, max_lookback: int = None
    ) -> list:
        """Generate PIT Entering or PIT Exiting Aisle End events."""
        # TODO: just iterate over tracklets and get the current actor by looking at the
        #       values
        events = []
        for (
            track_id,
            tracklet,
        ) in vignette.tracklets.items():
            if tracklet.category != ActorCategory.PIT:
                continue

            pit = tracklet.get_actor_at_timestamp(
                vignette.present_timestamp_ms
            )
            if pit is None:
                continue
            pit_track_id = track_id
            # First check if an actor is intersects an aisle_end. This indicates the
            # actor is currently on the side of the door away from the camera.
            pit_partially_contained_now = (
                aisle_end.get_shapely_polygon().intersects(
                    pit.get_shapely_polygon()
                )
            )

            stop_search = False
            # start from the most recent history and move backwards
            for index, historical_frame_struct in enumerate(
                vignette.past_frame_structs[::-1]
            ):
                if (max_lookback and index >= max_lookback) or stop_search:
                    break
                for historical_actor in historical_frame_struct.actors:
                    if historical_actor.track_id == pit_track_id:
                        pit_partially_contained_in_past = (
                            aisle_end.get_shapely_polygon().intersects(
                                historical_actor.get_shapely_polygon()
                            )
                        )
                        if (
                            pit_partially_contained_in_past
                            != pit_partially_contained_now
                        ):
                            event_type = (
                                EventType.PIT_EXITING_AISLE
                                if pit_partially_contained_now
                                else EventType.PIT_ENTERING_AISLE
                            )
                            events.append(
                                Event(
                                    timestamp_ms=vignette.present_frame_struct.relative_timestamp_ms,
                                    camera_uuid=self._camera_uuid,
                                    subject_id=str(pit.track_id),
                                    subject_uuid=pit.track_uuid,
                                    event_type=event_type,
                                    object_id=str(aisle_end.track_id),
                                    object_uuid=aisle_end.track_uuid,
                                    end_timestamp_ms=self._get_end_timestamp_ms(
                                        vignette
                                    ),
                                    x_velocity_pixel_per_sec=tracklet.x_velocity_pixel_per_sec,
                                    y_velocity_pixel_per_sec=tracklet.x_velocity_pixel_per_sec,
                                    normalized_speed=tracklet.normalized_pixel_speed,
                                )
                            )
                        stop_search = True
        return events

    def _get_events(self, vignette: Vignette) -> list:
        events = []
        for actor in vignette.present_frame_struct.actors:
            # Check intersection cross
            if actor.category == ActorCategory.INTERSECTION:
                events.extend(
                    self._get_pit_crossing_events_for_intersection(
                        actor, vignette
                    )
                )

            # Check aisle end cross
            if actor.category == ActorCategory.AISLE_END:
                events.extend(
                    self._get_pit_crossing_events_for_aisle_end(
                        actor, vignette
                    )
                )
        return events

    def process_vignette(self, vignette: Vignette) -> GeneratorResponse:
        states = self._get_states(vignette)
        events = self._get_events(vignette)
        return GeneratorResponse(events=events, states=states)
