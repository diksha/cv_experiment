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

import attr  # trunk-ignore(mypy/import,mypy/note)

from core.structs.actor import ActorCategory, DoorState
from core.structs.ergonomics import PostureType
from core.structs.protobufs.v1.actor_pb2 import (
    ActorCategory as ActorCategoryPb,
)
from core.structs.protobufs.v1.actor_pb2 import PostureType as PostureTypePb
from core.structs.protobufs.v1.state_pb2 import State as StatePb

# trunk-ignore-all(pylint/E0611,pylint/R0912): ignore pb import errors and too many branch warnings


@attr.s(slots=True)
class State:

    timestamp_ms: int = attr.ib()
    camera_uuid: str = attr.ib()
    actor_id: str = attr.ib()
    actor_category: ActorCategory = attr.ib()
    end_timestamp_ms: typing.Optional[int] = attr.ib(default=None)
    run_uuid: typing.Optional[str] = attr.ib(default=None)

    door_is_open: typing.Optional[bool] = attr.ib(default=None)
    door_state: typing.Optional[DoorState] = attr.ib(default=None)
    motion_zone_is_in_motion: typing.Optional[bool] = attr.ib(default=None)
    person_lift_type: typing.Optional[PostureType] = attr.ib(default=None)
    person_reach_type: typing.Optional[PostureType] = attr.ib(default=None)
    person_is_wearing_safety_vest: typing.Optional[bool] = attr.ib(
        default=None
    )
    person_is_wearing_hard_hat: typing.Optional[bool] = attr.ib(default=None)
    person_is_carrying_object: typing.Optional[bool] = attr.ib(default=None)

    pit_is_stationary: typing.Optional[bool] = attr.ib(default=None)
    obstruction_is_stationary: typing.Optional[bool] = attr.ib(default=None)
    person_is_associated: typing.Optional[bool] = attr.ib(default=None)
    person_in_no_ped_zone: typing.Optional[bool] = attr.ib(default=None)
    pit_in_driving_area: typing.Optional[bool] = attr.ib(default=None)
    pit_is_associated: typing.Optional[bool] = attr.ib(default=None)
    num_persons_in_no_ped_zone: typing.Optional[int] = attr.ib(default=None)
    track_uuid: typing.Optional[int] = attr.ib(default=None)

    @property
    def grouping_key(self) -> tuple:
        return (
            self.__class__.__name__,
            self.actor_category.value,
            self.actor_id,
        )

    @property
    # trunk-ignore(pylint/R0911)
    def differentiator(self) -> typing.Any:
        if self.actor_category == ActorCategory.MOTION_DETECTION_ZONE:
            return self.motion_zone_is_in_motion
        if self.actor_category == ActorCategory.NO_PED_ZONE:
            return self.num_persons_in_no_ped_zone
        if self.actor_category == ActorCategory.DOOR:
            return self.door_state
        if self.actor_category == ActorCategory.PERSON:
            return (
                self.person_lift_type,
                self.person_reach_type,
                self.person_is_wearing_hard_hat,
                self.person_is_wearing_safety_vest,
                self.person_is_associated,
                self.person_in_no_ped_zone,
                self.person_is_carrying_object,
            )
        if self.actor_category == ActorCategory.PIT:
            return (
                self.pit_is_stationary,
                self.pit_in_driving_area,
                self.pit_is_associated,
            )
        if self.actor_category == ActorCategory.OBSTRUCTION:
            return self.obstruction_is_stationary
        return None

    def to_proto(self) -> StatePb:
        """Converts the State to state.proto

        Returns:
            StatePb: converted state to protobuf
        """
        proto = StatePb()
        proto.timestamp_ms = self.timestamp_ms
        proto.camera_uuid = self.camera_uuid
        proto.actor_id = self.actor_id
        proto.actor_category = ActorCategoryPb.Value(
            f"ACTOR_CATEGORY_{self.actor_category.name}"
        )
        proto.end_timestamp_ms = self.end_timestamp_ms

        if self.run_uuid is not None:
            proto.run_uuid = self.run_uuid

        if self.door_is_open is not None:
            proto.door_is_open = self.door_is_open

        if self.person_lift_type is not None:
            proto.person_lift_type = PostureTypePb.Value(
                f"POSTURE_TYPE_{self.person_lift_type.name}"
            )
        if self.person_reach_type is not None:
            proto.person_reach_type = PostureTypePb.Value(
                f"POSTURE_TYPE_{self.person_reach_type.name}"
            )

        if self.person_is_wearing_safety_vest is not None:
            proto.person_is_wearing_safety_vest = (
                self.person_is_wearing_safety_vest
            )

        if self.person_is_wearing_hard_hat is not None:
            proto.person_is_wearing_hard_hat = self.person_is_wearing_hard_hat

        if self.person_is_carrying_object is not None:
            proto.person_is_carrying_object = self.person_is_carrying_object

        if self.pit_is_stationary is not None:
            proto.pit_is_stationary = self.pit_is_stationary

        if self.obstruction_is_stationary is not None:
            proto.obstruction_is_stationary = self.obstruction_is_stationary

        if self.person_is_associated is not None:
            proto.person_is_associated = self.person_is_associated

        if self.person_in_no_ped_zone is not None:
            proto.person_in_no_ped_zone = self.person_in_no_ped_zone

        if self.pit_in_driving_area is not None:
            proto.pit_in_driving_area = self.pit_in_driving_area

        if self.pit_is_associated is not None:
            proto.pit_is_associated = self.pit_is_associated

        if self.motion_zone_is_in_motion is not None:
            proto.motion_zone_is_in_motion = self.motion_zone_is_in_motion

        if self.num_persons_in_no_ped_zone is not None:
            proto.num_persons_in_no_ped_zone = self.num_persons_in_no_ped_zone

        if self.door_state is not None:
            proto.door_state = self.door_state.value

        if self.track_uuid is not None:
            proto.track_uuid = self.track_uuid

        return proto

    @classmethod
    def from_proto(cls, proto: StatePb) -> "State":
        """Converts the State from state.proto

        Args:
            proto (StatePb): protobuf state message

        Returns:
            State: converted state from protobuf
        """
        state = State(
            timestamp_ms=proto.timestamp_ms,
            camera_uuid=proto.camera_uuid,
            actor_id=proto.actor_id,
            actor_category=ActorCategory[
                ActorCategoryPb.Name(proto.actor_category).replace(
                    "ACTOR_CATEGORY_", ""
                )
            ],
        )

        state.end_timestamp_ms = proto.end_timestamp_ms
        state.track_uuid = proto.track_uuid

        # we do some weird stuff to find the optional fields
        # optionals are implemented as oneofs by the protobuf library
        for oneof_name in StatePb.DESCRIPTOR.oneofs_by_name:
            if not oneof_name.startswith("_"):
                # not sure to do with real oneof fields
                # they probably need to be handled specially below
                continue

            name = oneof_name[1:]
            if not proto.HasField(name):
                # field is unset, skip it
                continue

            if StatePb.DESCRIPTOR.fields_by_name[name].GetOptions().deprecated:
                # skip deprecated fields
                continue

            if name in ["person_lift_type", "person_reach_type"]:
                setattr(
                    state,
                    name,
                    PostureType[
                        PostureTypePb.Name(getattr(proto, name)).replace(
                            "POSTURE_TYPE_", ""
                        )
                    ],
                )
            elif name == "door_state":
                state.door_state = DoorState(proto.door_state)
            else:
                setattr(state, name, getattr(proto, name))

        return state
