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

from datetime import datetime, timezone
from typing import Dict

from google.cloud import pubsub_v1
from loguru import logger

from core.portal.state.models.event import Event as EventModel
from core.portal.state.models.state import State as StateModel
from core.structs.actor import ActorCategory as ActorCategoryEnum
from core.structs.ergonomics import ActivityType as ActivityTypeEnum
from core.structs.ergonomics import PostureType as PostureTypeEnum
from core.structs.event import EventType as EventTypeEnum
from core.structs.protobufs.v1.actor_pb2 import ActivityType as ActivityTypePb
from core.structs.protobufs.v1.actor_pb2 import (
    ActorCategory as ActorCategoryPb,
)
from core.structs.protobufs.v1.actor_pb2 import PostureType as PostureTypePb
from core.structs.protobufs.v1.event_pb2 import Event as EventPb
from core.structs.protobufs.v1.event_pb2 import EventType as EventTypePb
from core.structs.protobufs.v1.state_pb2 import State as StatePb


def get_common_fields(camera_uuid: str) -> Dict[str, str]:
    results = camera_uuid.split("/")
    return {
        "organization": results[0],
        "location": results[1],
        "zone": results[2],
        "camera_name": results[3],
    }


def event_message_callback(
    message: pubsub_v1.subscriber.message.Message,
) -> None:
    event = EventPb()
    event.ParseFromString(message.data)

    event_type_name = EventTypePb.Name(event.event_type).replace(
        "EVENT_TYPE_", ""
    )
    event_type = EventTypeEnum[event_type_name].value

    # Write to timescale
    EventModel.objects.create(
        # We do the following to avoid not losing precision with floating point division.
        timestamp=datetime.utcfromtimestamp(event.timestamp_ms // 1000)
        .replace(microsecond=(event.timestamp_ms % 1000) * 1000)
        .replace(tzinfo=timezone.utc),
        camera_uuid=event.camera_uuid,
        subject_id=event.subject_id if event.HasField("subject_id") else None,
        object_id=event.object_id if event.HasField("object_id") else None,
        event_type=event_type,
        end_timestamp=datetime.utcfromtimestamp(event.end_timestamp_ms // 1000)
        .replace(microsecond=(event.end_timestamp_ms % 1000) * 1000)
        .replace(tzinfo=timezone.utc),
        run_uuid=event.run_uuid if event.HasField("run_uuid") else None,
        x_velocity_pixel_per_sec=event.x_velocity_pixel_per_sec
        if event.HasField("x_velocity_pixel_per_sec")
        else None,
        y_velocity_pixel_per_sec=event.y_velocity_pixel_per_sec
        if event.HasField("y_velocity_pixel_per_sec")
        else None,
        normalized_speed=event.normalized_speed
        if event.HasField("normalized_speed")
        else None,
        **get_common_fields(event.camera_uuid)
    )

    # Acknowledge the message as processed.
    message.ack()


def state_message_callback(
    message: pubsub_v1.subscriber.message.Message,
) -> None:
    logger.info(message.data)
    state = StatePb()
    state.ParseFromString(message.data)

    actor_category_name = ActorCategoryPb.Name(state.actor_category).replace(
        "ACTOR_CATEGORY_", ""
    )
    actor_category = ActorCategoryEnum[actor_category_name].value

    if state.HasField("person_activity_type"):
        person_activity_type_name = ActivityTypePb.Name(
            state.person_activity_type
        ).replace("ACTIVITY_TYPE_", "")
        person_activity_type = ActivityTypeEnum[
            person_activity_type_name
        ].value
    else:
        person_activity_type = None

    if state.HasField("person_posture_type"):
        person_posture_type_name = PostureTypePb.Name(
            state.person_posture_type
        ).replace("POSTURE_TYPE_", "")
        person_posture_type = PostureTypeEnum[person_posture_type_name].value
    else:
        person_posture_type = None

    if state.HasField("person_lift_type"):
        person_lift_type_name = PostureTypePb.Name(
            state.person_lift_type
        ).replace("POSTURE_TYPE_", "")
        person_lift_type = PostureTypeEnum[person_lift_type_name].value
    else:
        person_lift_type = None

    if state.HasField("person_reach_type"):
        person_reach_type_name = PostureTypePb.Name(
            state.person_reach_type
        ).replace("POSTURE_TYPE_", "")
        person_reach_type = PostureTypeEnum[person_reach_type_name].value
    else:
        person_reach_type = None

    # Write to timescale
    StateModel.objects.create(
        # We do the following to avoid not losing precision with floating point division.
        timestamp=datetime.utcfromtimestamp(state.timestamp_ms // 1000)
        .replace(microsecond=(state.timestamp_ms % 1000) * 1000)
        .replace(tzinfo=timezone.utc),
        camera_uuid=state.camera_uuid,
        actor_id=state.actor_id,
        actor_category=actor_category,
        end_timestamp=datetime.utcfromtimestamp(state.end_timestamp_ms // 1000)
        .replace(microsecond=(state.end_timestamp_ms % 1000) * 1000)
        .replace(tzinfo=timezone.utc),
        run_uuid=state.run_uuid if state.HasField("run_uuid") else None,
        door_is_open=state.door_is_open
        if state.HasField("door_is_open")
        else None,
        person_activity_type=person_activity_type,
        person_posture_type=person_posture_type,
        person_lift_type=person_lift_type,
        person_reach_type=person_reach_type,
        person_is_wearing_safety_vest=state.person_is_wearing_safety_vest
        if state.HasField("person_is_wearing_safety_vest")
        else None,
        person_is_wearing_hard_hat=state.person_is_wearing_hard_hat
        if state.HasField("person_is_wearing_hard_hat")
        else None,
        person_is_carrying_object=state.person_is_carrying_object
        if state.HasField("person_is_carrying_object")
        else None,
        pit_is_stationary=state.pit_is_stationary
        if state.HasField("pit_is_stationary")
        else None,
        person_is_associated=state.person_is_associated
        if state.HasField("person_is_associated")
        else None,
        pit_in_driving_area=state.pit_in_driving_area
        if state.HasField("pit_in_driving_area")
        else None,
        person_in_no_ped_zone=state.person_in_no_ped_zone
        if state.HasField("person_in_no_ped_zone")
        else None,
        pit_is_associated=state.pit_is_associated
        if state.HasField("pit_is_associated")
        else None,
        motion_zone_is_in_motion=state.motion_zone_is_in_motion
        if state.HasField("motion_zone_is_in_motion")
        else None,
        **get_common_fields(state.camera_uuid)
    )

    # Acknowledge the message as processed.
    message.ack()
