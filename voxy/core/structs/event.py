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
from enum import Enum

import attr

from core.structs.protobufs.v1.event_pb2 import Event as EventPb
from core.structs.protobufs.v1.event_pb2 import EventType as EventTypePb


class EventType(Enum):
    UNKNOWN = 0
    DOOR_OPENED = 1
    DOOR_CLOSED = 2
    PIT_ENTERING_DOOR = 3
    PIT_EXITING_DOOR = 4
    DOOR_PARTIALLY_OPENED = 5
    PIT_ENTERING_INTERSECTION = 6
    PIT_EXITING_INTERSECTION = 7
    PIT_ENTERING_AISLE = 8
    PIT_EXITING_AISLE = 9
    PERSON_ENTERING_DOOR = 10
    PERSON_EXITING_DOOR = 11
    SPILL_DETECTED = 12


@attr.s(slots=True)
class Event:

    timestamp_ms: float = attr.ib()
    camera_uuid: str = attr.ib()
    subject_id: typing.Optional[str] = attr.ib()
    event_type: EventType = attr.ib()

    object_id: str = attr.ib()
    end_timestamp_ms: int = attr.ib(default=None)
    run_uuid: typing.Optional[str] = attr.ib(default=None)
    x_velocity_pixel_per_sec: typing.Optional[float] = attr.ib(default=None)
    y_velocity_pixel_per_sec: typing.Optional[float] = attr.ib(default=None)
    normalized_speed: typing.Optional[float] = attr.ib(default=None)

    subject_uuid: typing.Optional[str] = attr.ib(default=None)
    object_uuid: typing.Optional[str] = attr.ib(default=None)

    @property
    def grouping_key(self) -> tuple:
        return (self.event_type.value, self.subject_id, self.object_id)

    @property
    def differentiator(self) -> tuple:
        return self.grouping_key

    def to_proto(self):
        proto = EventPb()
        proto.timestamp_ms = self.timestamp_ms
        proto.camera_uuid = self.camera_uuid
        proto.event_type = EventTypePb.Value(
            f"EVENT_TYPE_{self.event_type.name}"
        )
        proto.end_timestamp_ms = self.end_timestamp_ms

        if self.run_uuid is not None:
            proto.run_uuid = self.run_uuid

        if self.subject_id is None and self.object_id is None:
            raise RuntimeError("Both Subject and Object ID can't be None")

        if self.subject_id is not None:
            proto.subject_id = self.subject_id

        if self.object_id is not None:
            proto.object_id = self.object_id

        if self.x_velocity_pixel_per_sec is not None:
            proto.x_velocity_pixel_per_sec = self.x_velocity_pixel_per_sec

        if self.y_velocity_pixel_per_sec is not None:
            proto.y_velocity_pixel_per_sec = self.y_velocity_pixel_per_sec

        if self.normalized_speed is not None:
            proto.normalized_speed = self.normalized_speed

        if self.subject_uuid is not None:
            proto.subject_uuid = self.subject_uuid

        if self.object_uuid is not None:
            proto.object_uuid = self.object_uuid

        return proto

    @classmethod
    def from_proto(cls, proto: EventPb) -> "Event":
        """Constructs an Event message from an event protobuf

        Args:
            proto (EventPb): protobuf event message

        Returns:
            Event: translated event object
        """
        event = Event(
            timestamp_ms=proto.timestamp_ms,
            camera_uuid=proto.camera_uuid,
            subject_id=None,  # these will be set later
            object_id=None,  # these will be set later
            event_type=EventType[
                EventTypePb.Name(proto.event_type).replace("EVENT_TYPE_", "")
            ],
        )

        event.end_timestamp_ms = proto.end_timestamp_ms

        # we do some weird stuff to find the optional fields
        # optionals are implemented as oneofs by the protobuf library
        for oneof_name in EventPb.DESCRIPTOR.oneofs_by_name:
            if not oneof_name.startswith("_"):
                # not sure to do with real oneof fields
                # they probably need to be handled specially below
                continue

            name = oneof_name[1:]
            if not proto.HasField(name):
                # field is unset, skip it
                continue

            if EventPb.DESCRIPTOR.fields_by_name[name].GetOptions().deprecated:
                # skip deprecated fields
                continue

            setattr(event, name, getattr(proto, name))

        return event
