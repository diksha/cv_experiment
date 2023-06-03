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
# trunk-ignore-all(pylint/C0302)

import hashlib
import typing
import uuid
from enum import Enum
from typing import Optional

import attr
import cv2
from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.ImageAnnotations_pb2 import ImageAnnotations
from foxglove_schemas_protobuf.Point2_pb2 import Point2
from foxglove_schemas_protobuf.PointsAnnotation_pb2 import PointsAnnotation
from foxglove_schemas_protobuf.TextAnnotation_pb2 import TextAnnotation
from google.protobuf.json_format import MessageToDict
from shapely.geometry import Polygon as ShapelyPolygon

from core.common.utils.proto_utils import VoxelProto
from core.structs.attributes import Polygon, Pose
from core.structs.ergonomics import Activity, ActivityType, PostureType

# trunk-ignore-begin(pylint/W0611)
# trunk-ignore-begin(flake8/F401)
# trunk-ignore-begin(pylint/E0611)
from protos.perception.structs.v1.actor_pb2 import (
    ActivityType as ActivityTypePb,
)
from protos.perception.structs.v1.actor_pb2 import Actor as ActorPb
from protos.perception.structs.v1.actor_pb2 import (
    ActorCategory as ActorCategoryPb,
)
from protos.perception.structs.v1.actor_pb2 import AisleActor as AisleActorPb
from protos.perception.structs.v1.actor_pb2 import (
    AisleImmutableAttributes as AisleImmutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import (
    AisleMutableAttributes as AisleMutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import DoorActor as DoorActorPb
from protos.perception.structs.v1.actor_pb2 import (
    DoorImmutableAttributes as DoorImmutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import (
    DoorMutableAttributes as DoorMutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import (
    DoorOrientation as DoorOrientationPb,
)
from protos.perception.structs.v1.actor_pb2 import DoorState as DoorStatePb
from protos.perception.structs.v1.actor_pb2 import DoorType as DoorTypePb
from protos.perception.structs.v1.actor_pb2 import (
    ErgonomicSeverityMetrics as ErgonomicSeverityMetricsPb,
)
from protos.perception.structs.v1.actor_pb2 import (
    HeadCoveringType as HeadCoveringTypePb,
)
from protos.perception.structs.v1.actor_pb2 import (
    IntersectionActor as IntersectionActorPb,
)
from protos.perception.structs.v1.actor_pb2 import (
    IntersectionImmutableAttributes as IntersectionImmutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import (
    IntersectionMutableAttributes as IntersectionMutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import (
    MotionDetectionZoneState as MotionDetectionZoneStatePb,
)
from protos.perception.structs.v1.actor_pb2 import PersonActor as PersonActorPb
from protos.perception.structs.v1.actor_pb2 import (
    PersonImmutableAttributes as PersonImmutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import (
    PersonMutableAttributes as PersonMutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import PitActor as PitActorPb
from protos.perception.structs.v1.actor_pb2 import (
    PitImmutableAttributes as PitImmutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import PitType as PitTypePb
from protos.perception.structs.v1.actor_pb2 import PostureType as PostureTypePb
from protos.perception.structs.v1.actor_pb2 import (
    ProductionLineActor as ProductionLineActorPb,
)
from protos.perception.structs.v1.actor_pb2 import (
    ProductionLineImmutableAttributes as ProductionLineImmutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import (
    ProductionLineMutableAttributes as ProductionLineMutableAttributesPb,
)
from protos.perception.structs.v1.actor_pb2 import Skeleton as SkeletonPb

# trunk-ignore-end(pylint/E0611)
# trunk-ignore-end(flake8/F401)
# trunk-ignore-end(pylint/W0611)


# DO NOT CHANGE OR MODIFY ENUMERATIONS
class ActorCategory(Enum):
    UNKNOWN = 0
    PERSON = 1
    PIT = 2
    DOOR = 3
    HARD_HAT = 4
    SAFETY_VEST = 5
    BARE_CHEST = 6
    BARE_HEAD = 7
    INTERSECTION = 8
    AISLE_END = 9
    PERSON_V2 = 10
    PIT_V2 = 11
    NO_PED_ZONE = 12
    DRIVING_AREA = 13
    TRUCK = 14
    VEHICLE = 15
    TRAILER = 16
    BIKE = 17
    BUS = 18
    MOTION_DETECTION_ZONE = 19
    SAFETY_GLOVE = 20
    BARE_HAND = 21
    SPILL = 22
    PALLET = 23
    COVERED_HEAD = 24
    OBSTRUCTION = 25


class DoorState(Enum):
    FULLY_OPEN = 0
    FULLY_CLOSED = 1
    PARTIALLY_OPEN = 2
    UNKNOWN = 3


class DoorType(Enum):
    DOCK = 0
    FREEZER = 1
    EXIT = 2
    UNKNOWN = 3
    CURTAIN = 4


class DoorOrientation(Enum):
    FRONT_DOOR = (0,)
    SIDE_DOOR = 0


class OccludedDegree(Enum):
    """Degree of how much an actor is occluded.

    Occluded: <80%
    HeavilyOccluded: >80%
    FullyOccluded: Not seen at all
    """

    NONE = 0
    Occluded = 1
    HeavilyOccluded = 2
    FullyOccluded = 3


class OperatingObject(Enum):
    """What object person is operating

    Occluded: <20%
    HeavilyOccluded: <80%
    FullyOccluded: >80%
    """

    NONE = 0
    PIT = 1
    VEHICLE = 2
    TRUCK = 3
    BIKE = 4


class MotionDetectionZoneState(Enum):
    """
    Motion detection zone classification
    """

    UNKNOWN = 0
    FROZEN = 1
    MOTION = 2


class HeadCoveringType(Enum):
    """
    Head covering type
    UNKNOWN: catch all
    LEGACY_BARE_HEAD: legacy support for is_wearing_hard_hat=False
    BARE_HEAD: no head covering
    COVERED_HEAD: any head covering not a hard hat
    HARD_HAT: hard hat head covering
    """

    UNKNOWN = 0
    LEGACY_BARE_HEAD = 1
    BARE_HEAD = 2
    COVERED_HEAD = 3
    HARD_HAT = 4


DOOR_TYPE_MAP = {
    "dock": DoorType.DOCK,
    "freezer": DoorType.FREEZER,
    "exit": DoorType.EXIT,
    "curtain": DoorType.CURTAIN,
    "unknown": DoorType.UNKNOWN,
}

DOOR_TYPE_PRIORITY = {
    DoorType.FREEZER: 5,
    DoorType.EXIT: 4,
    DoorType.DOCK: 3,
    DoorType.UNKNOWN: 2,
    DoorType.CURTAIN: 1,
}


def get_actor_id_from_actor_category_and_track_id(
    track_id: int, actor_category: ActorCategory
) -> int:
    """Returns actor id from track id and actor category.

    Args:
        track_id (int): track id of the actor
        actor_category (ActorCategory): category of the actor

    Returns:
        int: actor id
    """
    # TODO resolve the actor ID issue. 1000 is a magic number
    # to ensure actor_ids don't collide
    return track_id + 1000 * actor_category.value


def get_track_uuid(
    camera_uuid: str,
    unique_identifier: str,
    category: ActorCategory,
    video_name: Optional[str] = None,
    run_seed: Optional[str] = None,
) -> str:
    """Generates uuid given camera_uuid, unique_identifier and video_name if present.

    Note: Works with an assumption that unique_identifier chosen by the user is unique
    across category.

    Args:
        camera_uuid (str): uuid of the camera
        unique_identifier (str): unique identifier across a category and camera
        category (ActorCategory): Category of the actor
        video_name (str, optional): . Defaults to None.
        run_seed (str, optional): random uuid representing run instance

    Returns:
        str: unique identifier for the track of an actor
    """
    seed = "".join(
        filter(
            None,
            [
                camera_uuid,
                unique_identifier,
                category.name,
                video_name,
                run_seed,
            ],
        )
    )
    m = hashlib.new("md5", usedforsecurity=False)
    m.update(seed.encode("utf-8"))
    return str(uuid.UUID(m.hexdigest(), version=4))


def draw_text(
    img,
    text,
    font=cv2.FONT_HERSHEY_PLAIN,
    pos=(0, 0),
    font_scale=1,
    font_thickness=1,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
):

    x_origin, y_origin = pos
    # Ensure newline characters are properly applied
    for i, line in enumerate(text.split("\n")):
        text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        text_w, text_h = text_size
        y_old = y_origin if i == 0 else y_origin + (i - 1) * text_h * 1.5
        y_current = y_origin + i * text_h * 1.5
        y_old = int(y_old)
        y_current = int(y_current)
        cv2.rectangle(
            img,
            (x_origin, y_old),
            (x_origin + text_w, y_current + text_h),
            text_color_bg,
            -1,
        )
        cv2.putText(
            img,
            line,
            (x_origin, y_current),
            font,
            font_scale,
            text_color,
            font_thickness,
        )

    return text_size


def get_ordered_actors(
    actor_list: typing.List[str],
) -> typing.List[ActorCategory]:
    """Returns an ordered list of actors based on their order in ActorCategory
    Args:
        actor_list (List[str]): list of actor category names
    Returns:
        List[ActorCategory]: list of ActorCategory enums in their correct order
    """
    return [actor for actor in ActorCategory if actor.name in actor_list]


@attr.s(slots=True)
class Actor:

    # TODO enable hasher/comparator that takes in attribute names
    # and accordingly compare two actors.
    # It would be awesome have it built from the start.

    uuid = attr.ib(default=None, type=str)
    category = attr.ib(default=None, type=ActorCategory)
    polygon = attr.ib(default=None, type=Polygon)
    # TODO: deprecate in favor of track uuid which is of uuid4 format
    track_id = attr.ib(default=None, type=int)
    track_uuid = attr.ib(default=None, type=str)
    pose = attr.ib(default=None, type=Pose)
    confidence = attr.ib(default=None, type=float)
    manual = attr.ib(default=None, type=bool)

    occluded = attr.ib(default=None, type=bool)  # deprecated
    truncated = attr.ib(default=None, type=bool)

    occluded_degree = attr.ib(default=None, type=OccludedDegree)

    human_operating = attr.ib(default=None, type=bool)
    forklift = attr.ib(default=None, type=bool)
    loaded = attr.ib(default=None, type=bool)
    forks_raised = attr.ib(default=None, type=bool)

    operating_pit = attr.ib(default=None, type=bool)  # deprecated
    operating_object = attr.ib(default=None, type=OperatingObject)

    door_state = attr.ib(default=None, type=DoorState)
    door_type = attr.ib(default=None, type=DoorType)
    door_orientation = attr.ib(default=None, type=DoorOrientation)
    door_state_probabilities = attr.ib(default=None, type=list)

    x_velocity_pixel_per_sec = attr.ib(default=None, type=float)
    y_velocity_pixel_per_sec = attr.ib(default=None, type=float)
    normalized_pixel_speed = attr.ib(default=None, type=float)

    x_velocity_meters_per_sec = attr.ib(default=None, type=float)
    y_velocity_meters_per_sec = attr.ib(default=None, type=float)

    x_position_m = attr.ib(default=None, type=float)
    y_position_m = attr.ib(default=None, type=float)
    z_position_m = attr.ib(default=None, type=float)

    distance_to_camera_m = attr.ib(default=None, type=float)

    activity = attr.ib(default=None, type=dict)

    is_wearing_safety_vest = attr.ib(default=None, type=bool)
    is_pickup = attr.ib(default=None, type=bool)
    is_motorized = attr.ib(default=None, type=bool)
    is_van = attr.ib(default=None, type=bool)
    # This is used for labeling purposes
    is_wearing_safety_vest_v2 = attr.ib(default=None, type=bool)
    is_wearing_hard_hat = attr.ib(default=None, type=bool)
    motion_detection_zone_state = attr.ib(
        default=None, type=MotionDetectionZoneState
    )
    is_carrying_object = attr.ib(default=None, type=bool)
    motion_detection_score_std = attr.ib(default=None, type=float)
    head_covering_type = attr.ib(default=None, type=HeadCoveringType)
    skeleton = attr.ib(default=None, type=SkeletonPb)
    ergonomic_severity_metrics = attr.ib(
        default=None, type=ErgonomicSeverityMetricsPb
    )

    def get_shapely_polygon(self):
        return ShapelyPolygon([[p.x, p.y] for p in self.polygon.vertices])

    def get_box(self):
        min_x, min_y, max_x, max_y = 100000, 100000, 0, 0
        for vertice in self.polygon.vertices:
            min_x = min(vertice.x, min_x)
            min_y = min(vertice.y, min_y)
            max_x = max(vertice.x, max_x)
            max_y = max(vertice.y, max_y)
        return [min_x, min_y, max_x, max_y]

    def to_dict(self):
        activity_dict = None
        if self.activity:
            activity_dict = {
                activity_type: self.activity[activity_type].posture.name
                for activity_type in self.activity
            }
        head_covering_type = None
        if self.head_covering_type:
            head_covering_type = self.head_covering_type.name
        elif self.is_wearing_hard_hat is not None:
            head_covering_type = (
                HeadCoveringType.HARD_HAT.name
                if self.is_wearing_hard_hat
                else HeadCoveringType.LEGACY_BARE_HEAD.name
            )

        return {
            "uuid": self.uuid,
            "category": self.category.name if self.category else None,
            "polygon": self.polygon.to_dict() if self.polygon else None,
            "track_id": self.track_id,
            "track_uuid": self.track_uuid,
            "manual": self.manual,
            "occluded": self.occluded,
            "occluded_degree": self.occluded_degree.name
            if self.occluded_degree
            else None,
            "operating_object": self.operating_object.name
            if self.operating_object
            else None,
            "truncated": self.truncated,
            "confidence": self.confidence,
            "human_operating": self.human_operating,
            "forklift": self.forklift,
            "loaded": self.loaded,
            "forks_raised": self.forks_raised,
            "operating_pit": self.operating_pit,
            "door_state": self.door_state.name if self.door_state else None,
            "door_state_probabilities": self.door_state_probabilities,
            "door_type": self.door_type.name if self.door_type else None,
            "door_orientation": self.door_orientation.name
            if self.door_orientation
            else None,
            "pose": self.pose.to_dict() if self.pose else None,
            "activity": activity_dict,
            "x_velocity_pixel_per_sec": self.x_velocity_pixel_per_sec,
            "y_velocity_pixel_per_sec": self.y_velocity_pixel_per_sec,
            "x_velocity_meters_per_sec": self.x_velocity_meters_per_sec,
            "y_velocity_meters_per_sec": self.y_velocity_meters_per_sec,
            "x_position_m": self.x_position_m,
            "y_position_m": self.y_position_m,
            "z_position_m": self.z_position_m,
            "distance_to_camera_m": self.distance_to_camera_m,
            "is_wearing_safety_vest": self.is_wearing_safety_vest,
            "is_pickup": self.is_pickup,
            "is_motorized": self.is_motorized,
            "is_van": self.is_van,
            "is_wearing_safety_vest_v2": self.is_wearing_safety_vest_v2,
            "is_wearing_hard_hat": self.is_wearing_hard_hat,
            "motion_detection_zone_state": self.motion_detection_zone_state.name
            if self.motion_detection_zone_state
            else None,
            "is_carrying_object": self.is_carrying_object,
            "motion_detection_score_std": self.motion_detection_score_std
            if self.motion_detection_score_std is not None
            else None,
            "head_covering_type": head_covering_type,
            "ergonomic_severity_metrics": (
                MessageToDict(
                    self.ergonomic_severity_metrics,
                    preserving_proto_field_name=True,
                )
                if self.ergonomic_severity_metrics is not None
                else None
            ),
            "skeleton": (
                MessageToDict(self.skeleton, preserving_proto_field_name=True)
                if self.skeleton is not None
                else None
            ),
        }

    # trunk-ignore(pylint/R0915)
    def to_proto(self):
        """Converts the actor python class to actor.proto

        Returns:
            ActorPb: this actor converted to actor.proto
        """

        def proto_safe_set(
            proto_object: typing.Any, attribute_name: str, value: typing.Any
        ):
            """
            Safely sets protobuf attributes

            Args:
                proto_object (typing.Any): the current protobuf object to mutate
                attribute_name (str): the attribute to update
                value (typing.Any): the value of the attribute
            """
            if value is not None:
                setattr(proto_object, attribute_name, value)

        pose_proto = self.pose.to_proto() if self.pose else None
        kwargs = {}
        if pose_proto is not None:
            kwargs["pose"] = pose_proto
        kwargs["skeleton"] = self.skeleton
        kwargs["ergonomic_severity_metrics"] = self.ergonomic_severity_metrics
        person_attrs_pb = PersonMutableAttributesPb(**kwargs)
        person_pb = PersonActorPb(mutable_attributes=person_attrs_pb)
        proto_safe_set(
            person_pb.mutable_attributes,
            "is_wearing_hard_hat",
            self.is_wearing_hard_hat,
        )
        proto_safe_set(
            person_pb.mutable_attributes,
            "is_wearing_safety_vest",
            self.is_wearing_safety_vest,
        )
        proto_safe_set(
            person_pb.mutable_attributes,
            "is_carrying_object",
            self.is_carrying_object,
        )
        proto_safe_set(
            person_pb.mutable_attributes,
            "distance_to_camera_meters",
            self.distance_to_camera_m,
        )
        head_covering_type = None
        if self.head_covering_type:
            head_covering_type = HeadCoveringTypePb.Value(
                f"HEAD_COVERING_{self.head_covering_type.name}"
            )
        elif self.is_wearing_hard_hat is not None:
            head_covering_type = (
                HeadCoveringTypePb.Value("HEAD_COVERING_HARD_HAT")
                if self.is_wearing_hard_hat
                else HeadCoveringTypePb.Value("HEAD_COVERING_LEGACY_BARE_HEAD")
            )
        proto_safe_set(
            person_pb.mutable_attributes,
            "head_covering_type",
            head_covering_type,
        )

        pit_pb = PitActorPb()
        proto_safe_set(
            pit_pb.mutable_attributes,
            "is_human_operating",
            self.human_operating,
        )
        proto_safe_set(pit_pb.mutable_attributes, "is_loaded", self.loaded)
        proto_safe_set(
            pit_pb.mutable_attributes, "is_forks_raised", self.forks_raised
        )
        proto_safe_set(
            pit_pb.mutable_attributes.pixel_velocity,
            "x_velocity_pixels_per_second",
            self.x_velocity_pixel_per_sec,
        )
        proto_safe_set(
            pit_pb.mutable_attributes.pixel_velocity,
            "y_velocity_pixels_per_second",
            self.y_velocity_pixel_per_sec,
        )
        proto_safe_set(
            pit_pb.mutable_attributes,
            "normalized_pixel_speed_pixels_per_second",
            self.normalized_pixel_speed,
        )
        proto_safe_set(
            pit_pb.mutable_attributes.world_velocity,
            "x_velocity_meters_per_second",
            self.x_velocity_meters_per_sec,
        )
        proto_safe_set(
            pit_pb.mutable_attributes.world_velocity,
            "y_velocity_meters_per_second",
            self.y_velocity_meters_per_sec,
        )
        proto_safe_set(
            pit_pb.mutable_attributes.position,
            "x_position_meters",
            self.x_position_m,
        )
        proto_safe_set(
            pit_pb.mutable_attributes.position,
            "y_position_meters",
            self.y_position_m,
        )
        proto_safe_set(
            pit_pb.mutable_attributes.position,
            "z_position_meters",
            self.z_position_m,
        )
        proto_safe_set(
            pit_pb.mutable_attributes,
            "distance_to_camera_meters",
            self.distance_to_camera_m,
        )

        door_pb = DoorActorPb()
        proto_safe_set(
            door_pb.mutable_attributes,
            "state",
            DoorStatePb.Value(f"DOOR_STATE_{self.door_state.name}")
            if self.door_state
            else None,
        )
        if (
            self.door_state_probabilities
            and len(self.door_state_probabilities) == 3
        ):
            door_pb.mutable_attributes.state_probability.open_probability = (
                self.door_state_probabilities[0]
            )
            door_pb.mutable_attributes.state_probability.partially_open_probability = self.door_state_probabilities[  # trunk-ignore(pylint/C0301)
                1
            ]
            door_pb.mutable_attributes.state_probability.closed_probability = (
                self.door_state_probabilities[2]
            )

        proto_safe_set(
            door_pb.immutable_attributes,
            "type",
            DoorTypePb.Value(f"DOOR_TYPE_{self.door_type.name}")
            if self.door_type
            else None,
        )

        proto_safe_set(
            door_pb.immutable_attributes,
            "orientation",
            DoorOrientationPb.Value(
                f"DOOR_ORIENTATION_{self.door_orientation.name}"
            )
            if self.door_orientation
            else None,
        )

        intersection_pb = IntersectionActorPb()
        # intersection has no mutable or immutable attributes

        aisle_pb = AisleActorPb()
        # aisle also has no mutable or immutable attributes

        motion_detection_pb = ProductionLineActorPb()

        proto_safe_set(
            motion_detection_pb.mutable_attributes,
            "state",
            MotionDetectionZoneStatePb.Value(
                f"MOTION_DETECTION_ZONE_STATE_{self.motion_detection_zone_state.name}"
            )
            if self.motion_detection_zone_state
            else None,
        )

        proto_safe_set(
            motion_detection_pb.mutable_attributes,
            "score_std",
            self.motion_detection_score_std,
        )

        actor_map = {
            ActorCategory.PIT.name: ("pit_actor", pit_pb),
            ActorCategory.PERSON.name: ("person_actor", person_pb),
            ActorCategory.DOOR.name: ("door_actor", door_pb),
            ActorCategory.INTERSECTION.name: (
                "intersection_actor",
                intersection_pb,
            ),
            ActorCategory.AISLE_END.name: ("aisle_actor", aisle_pb),
            ActorCategory.MOTION_DETECTION_ZONE.name: (
                "production_line_actor",
                motion_detection_pb,
            ),
        }
        polygon = self.polygon.to_proto() if self.polygon else None

        actor_attributes = actor_map.get(self.category.name)
        actor_args = {
            "track_id": self.track_id,
            "confidence": self.confidence,
            "polygon": polygon,
            "category": ActorCategoryPb.Value(
                f"ACTOR_CATEGORY_{self.category.name}"
            ),
        }
        if self.uuid:
            actor_args["uuid"] = self.uuid
        if self.track_uuid:
            actor_args["track_uuid"] = self.track_uuid

        if actor_attributes is not None:
            attr_name, attribute = actor_attributes
            actor_args[attr_name] = attribute
        # Create Pb
        actor_pb = ActorPb(**actor_args)
        return actor_pb

    @classmethod
    # trunk-ignore(pylint/R0915)
    def from_proto(cls, raw_proto: ActorPb) -> "Actor":
        """
        Deserializes the protobuf struct and
        converts it to an actor struct

        Args:
            raw_proto (ActorPb): the current actor protobuf

        Returns:
            Actor: deserialized actor
        """

        proto = VoxelProto(raw_proto)
        # TODO: deserialize actor here
        kwargs = {}

        def update_args(name: str, item: typing.Any):
            """
            Updates the input arguments for the actor construction

            Args:
                name (str): the name of the argument
                item (typing.Any): the item to pass through arguments
            """
            if item is not None:
                kwargs[name] = item

        update_args("uuid", proto.uuid)
        update_args("track_uuid", proto.track_uuid)
        update_args("track_id", proto.track_id)
        update_args("track_uuid", proto.track_uuid)
        update_args("confidence", proto.confidence)
        update_args(
            "polygon",
            Polygon.from_proto(proto.polygon)
            if proto.polygon is not None
            else None,
        )
        update_args(
            "category",
            ActorCategory[
                ActorCategoryPb.Name(proto.category).replace(
                    "ACTOR_CATEGORY_", ""
                )
            ],
        )
        # TODO: add from proto for polygon

        def update_person_actor_attributes(person_actor: PersonActorPb):
            if person_actor.mutable_attributes is not None:
                update_args(
                    "is_wearing_hard_hat",
                    person_actor.mutable_attributes.is_wearing_hard_hat,
                )
                update_args(
                    "is_wearing_safety_vest",
                    person_actor.mutable_attributes.is_wearing_safety_vest,
                )
                update_args(
                    "is_carrying_object",
                    person_actor.mutable_attributes.is_carrying_object,
                )

                update_args(
                    "distance_to_camera_m",
                    person_actor.mutable_attributes.distance_to_camera_meters,
                )
                # TODO: add activity
                update_args(
                    "pose",
                    Pose.from_proto(person_actor.mutable_attributes.pose)
                    if VoxelProto(person_actor.mutable_attributes).pose
                    is not None
                    else None,
                )
                head_covering_type = None
                if person_actor.mutable_attributes.head_covering_type:
                    head_covering_type = HeadCoveringType[
                        HeadCoveringTypePb.Name(
                            person_actor.mutable_attributes.head_covering_type
                        ).replace("HEAD_COVERING_", "")
                    ]
                elif (
                    person_actor.mutable_attributes.is_wearing_hard_hat
                    is not None
                ):
                    head_covering_type = (
                        HeadCoveringType.HARD_HAT
                        if person_actor.mutable_attributes.is_wearing_hard_hat
                        else HeadCoveringType.LEGACY_BARE_HEAD
                    )
                update_args(
                    "head_covering_type",
                    head_covering_type,
                )
                update_args(
                    "skeleton",
                    person_actor.mutable_attributes.skeleton
                    if VoxelProto(person_actor.mutable_attributes).skeleton
                    is not None
                    else None,
                )
                update_args(
                    "ergonomic_severity_metrics",
                    person_actor.mutable_attributes.ergonomic_severity_metrics
                    if VoxelProto(
                        person_actor.mutable_attributes
                    ).ergonomic_severity_metrics
                    is not None
                    else None,
                )

        if proto.person_actor is not None:
            update_person_actor_attributes(proto.person_actor)

        def update_pit_actor_attributes(pit_actor: PitActorPb):
            attrs = pit_actor.mutable_attributes
            update_args(
                "x_velocity_pixel_per_sec",
                attrs.pixel_velocity.x_velocity_pixels_per_second,
            )
            update_args(
                "y_velocity_pixel_per_sec",
                attrs.pixel_velocity.y_velocity_pixels_per_second,
            )
            update_args(
                "x_velocity_meters_per_sec",
                attrs.world_velocity.x_velocity_meters_per_second,
            )
            update_args(
                "y_velocity_meters_per_sec",
                attrs.world_velocity.x_velocity_meters_per_second,
            )
            update_args("x_position_m", attrs.position.x_position_meters)
            update_args("y_position_m", attrs.position.y_position_meters)
            update_args("z_position_m", attrs.position.z_position_meters)
            update_args(
                "normalized_pixel_speed",
                attrs.normalized_pixel_speed_pixels_per_second,
            )
            update_args(
                "distance_to_camera_m",
                attrs.distance_to_camera_meters,
            )

        if proto.pit_actor is not None:
            update_pit_actor_attributes(proto.pit_actor)

        def update_door_actor_attributes(door_actor: DoorActorPb):
            mut_attrs = door_actor.mutable_attributes
            update_args(
                "door_state",
                DoorState[
                    DoorStatePb.Name(mut_attrs.state).replace(
                        "DOOR_STATE_", ""
                    )
                ]
                if mut_attrs.state
                else None,
            )
            update_args(
                "door_state_probabilities",
                [
                    mut_attrs.state_probability.open_probability,
                    mut_attrs.state_probability.partially_open_probability,
                    mut_attrs.state_probability.closed_probability,
                ]
                if mut_attrs.state_probability
                else None,
            )
            # TODO: see how we can rename the door state attributes

        if proto.door_actor is not None:
            update_door_actor_attributes(proto.door_actor)

        if proto.production_line_actor is not None:
            mut_attrs = VoxelProto(
                proto.production_line_actor.mutable_attributes
            )
            if mut_attrs.state is not None:
                update_args(
                    "motion_detection_zone_state",
                    MotionDetectionZoneState[
                        MotionDetectionZoneStatePb.Name(
                            mut_attrs.state
                        ).replace("MOTION_DETECTION_ZONE_STATE_", "")
                    ],
                )
            if mut_attrs.score_std is not None:
                update_args("motion_detection_score_std", mut_attrs.score_std)

        # aisle and intersection do not have mutable or immutable attributes
        return Actor(**kwargs)

    def to_annotation_protos(self) -> PointsAnnotation:
        """
        Generates the annotation protobuf for visualization

        Returns:
            PointsAnnotation: the current bounding box of the actor
        """
        points = [
            Point2(x=point.x, y=point.y) for point in self.polygon.vertices
        ]
        color_map = {
            "PERSON": Color(r=0, g=0.2, b=0.8, a=0.5),
            "PIT": Color(r=1, g=0.2, b=0.0, a=0.5),
            "SPILL": Color(r=0, g=1, b=0, a=0.5),
        }
        bounding_box = PointsAnnotation(
            type=2,
            points=points,
            outline_color=color_map[self.category.name]
            if self.category.name in color_map
            else Color(r=1.0, g=1.0, b=1.0, a=0.5),
            thickness=4,
        )
        return bounding_box

    def has_pose(self) -> bool:
        """
        Returns whether the actor has a pose

        Returns:
            bool: the pose
        """
        return self.pose is not None

    def to_pose_annotation_protos(self) -> ImageAnnotations:
        """
        Returns the pose annotation proto

        Returns:
            bool: the pose
        """
        return self.pose.to_annotation_protos()

    def to_text_annotation_protos(self) -> TextAnnotation:
        """
        Generates image annotations for viewing the objects in foxglove

        Returns:
            TextAnnotation: the image text annotation
        """

        text = TextAnnotation(
            position=Point2(
                x=self.polygon.vertices[0].x,
                y=self.polygon.vertices[0].y,
            ),
            text=f"[{self.category.name}: {self.track_id}]",
            font_size=7,
            text_color=Color(r=1.0, g=1.0, b=1.0, a=0.7),
            background_color=Color(r=0.0, g=0.0, b=0.0, a=0.7),
        )
        return text

    @staticmethod
    def get_head_covering_type_from_actor_dict(
        actor_data: typing.Dict[str, object]
    ) -> typing.Optional[HeadCoveringType]:
        """Helper function to get head covering type from actor dictionary
        Args:
            actor_data (Dict[str, object]): actor data dictionary
        Returns:
            Optional[HeadCoveringType]: head covering type if it exists
        """
        head_covering_type = None
        if actor_data.get("head_covering_type"):
            head_covering_type = HeadCoveringType[
                actor_data["head_covering_type"]
            ]
        elif actor_data.get("is_wearing_hard_hat") is not None:
            head_covering_type = (
                HeadCoveringType.HARD_HAT
                if actor_data["is_wearing_hard_hat"]
                else HeadCoveringType.LEGACY_BARE_HEAD
            )
        return head_covering_type

    @classmethod
    def from_metaverse(self, data):
        # TODO (Nasha) : Deal with new activitiy dictionary
        activity_dict = None
        if data.get("activity") is not None and (
            data["activity"].get(ActivityType.LIFTING.name) is not None
            or data["activity"].get(ActivityType.REACHING.name) is not None
        ):
            activity_dict = {
                activity_type: Activity(
                    ActivityType[activity_type],
                    PostureType[data["activity"][activity_type]],
                )
                for activity_type in data["activity"]
            }
        head_covering_type = self.get_head_covering_type_from_actor_dict(data)
        return Actor(
            uuid=data.get("uuid"),
            category=ActorCategory[data["category"]]
            if data.get("category")
            else None,
            polygon=Polygon.from_metaverse(data["polygon"])
            if data.get("polygon")
            else None,
            track_id=data.get("track_id"),
            track_uuid=data.get("track_uuid"),
            manual=data.get("manual"),
            occluded=data.get("occluded"),
            occluded_degree=OccludedDegree[data.get("occluded_degree")]
            if data.get("occluded_degree")
            else None,
            operating_object=OperatingObject[data.get("operating_object")]
            if data.get("operating_object")
            else None,
            truncated=data.get("truncated"),
            confidence=data.get("confidence"),
            human_operating=data.get("human_operating"),
            forklift=data.get("forklift"),
            loaded=data.get("loaded"),
            forks_raised=data.get("forks_raised"),
            operating_pit=data.get("operating_pit"),
            door_state=DoorState[data.get("door_state")]
            if data.get("door_state") is not None
            else None,
            door_state_probabilities=data.get("door_state_probabilities"),
            door_type=DoorType[data.get("door_type")]
            if data.get("door_type") is not None
            else None,
            door_orientation=DoorOrientation[data.get("door_orientation")]
            if data.get("door_orientation") is not None
            else None,
            pose=Pose.from_dict(data["pose"]) if data.get("pose") else None,
            activity=activity_dict,
            x_velocity_pixel_per_sec=data.get("x_velocity_pixel_per_sec"),
            y_velocity_pixel_per_sec=data.get("y_velocity_pixel_per_sec"),
            is_wearing_safety_vest=data.get("is_wearing_safety_vest"),
            is_wearing_safety_vest_v2=data.get("is_wearing_safety_vest_v2"),
            is_carrying_object=data.get("is_carrying_object"),
            is_pickup=data.get("is_pickup"),
            is_motorized=data.get("is_motorized"),
            is_van=data.get("is_van"),
            is_wearing_hard_hat=data.get("is_wearing_hard_hat"),
            motion_detection_zone_state=MotionDetectionZoneState[
                data.get("motion_detection_zone_state")
            ]
            if data.get("motion_detection_zone_state") is not None
            else None,
            head_covering_type=head_covering_type,
            skeleton=SkeletonPb(**data["skeleton"])
            if data.get("skeleton")
            else None,
            ergonomic_severity_metrics=ErgonomicSeverityMetricsPb(
                **data["ergonomic_severity_metrics"]
            )
            if data.get("ergonomic_severity_metrics")
            else None,
        )

    @classmethod
    def from_dict(self, data):
        if data.get("activity") is not None and (
            data["activity"].get(ActivityType.LIFTING.name) is not None
            or data["activity"].get(ActivityType.REACHING.name) is not None
        ):
            activity_dict = {
                activity_type: Activity(
                    ActivityType[activity_type],
                    PostureType[data["activity"][activity_type]],
                )
                for activity_type in data["activity"]
            }
        head_covering_type = self.get_head_covering_type_from_actor_dict(data)
        return Actor(  # trunk-ignore(mypy/call-arg)
            uuid=data.get("uuid"),
            category=ActorCategory[data["category"]]
            if data.get("category")
            else None,
            polygon=Polygon.from_dict(data["polygon"])
            if data.get("polygon")
            else None,
            track_id=data.get("track_id"),
            track_uuid=data.get("track_uuid"),
            manual=data.get("manual"),
            occluded=data.get("occluded"),
            occluded_degree=OccludedDegree[data.get("occluded_degree")]
            if data.get("occluded_degree")
            else None,
            operating_object=OperatingObject[
                data.get("operating_object", "NONE").upper()
            ]
            if data.get("operating_object")
            else None,
            truncated=data.get("truncated"),
            confidence=data.get("confidence"),
            human_operating=data.get("human_operating"),
            forklift=data.get("forklift"),
            loaded=data.get("loaded"),
            forks_raised=data.get("forks_raised"),
            operating_pit=data.get("operating_pit"),
            door_state=DoorState[data.get("door_state")]
            if data.get("door_state") is not None
            else None,
            door_state_probabilities=data.get("door_state_probabilities"),
            door_type=DoorType[data.get("door_type")]
            if data.get("door_type") is not None
            else None,
            door_orientation=DoorOrientation[data.get("door_orientation")]
            if data.get("door_orientation") is not None
            else None,
            pose=Pose.from_dict(data["pose"]) if data.get("pose") else None,
            activity=activity_dict
            if data.get("activity") is not None
            and (
                data["activity"].get(ActivityType.LIFTING.name) is not None
                or data["activity"].get(ActivityType.REACHING.name) is not None
            )
            else None,
            x_velocity_pixel_per_sec=data.get("x_velocity_pixel_per_sec"),
            y_velocity_pixel_per_sec=data.get("y_velocity_pixel_per_sec"),
            x_velocity_meters_per_sec=data.get("x_velocity_meters_per_sec"),
            y_velocity_meters_per_sec=data.get("y_velocity_meters_per_sec"),
            x_position_m=data.get("x_position_m"),
            y_position_m=data.get("y_position_m"),
            z_position_m=data.get("z_position_m"),
            distance_to_camera_m=data.get("distance_to_camera_m"),
            is_wearing_safety_vest=data.get("is_wearing_safety_vest"),
            is_wearing_safety_vest_v2=data.get("is_wearing_safety_vest_v2"),
            is_carrying_object=data.get("is_carrying_object"),
            is_pickup=data.get("is_pickup"),
            is_motorized=data.get("is_motorized"),
            is_van=data.get("is_van"),
            is_wearing_hard_hat=data.get("is_wearing_hard_hat"),
            motion_detection_zone_state=MotionDetectionZoneState[
                data.get(
                    "motion_detection_zone_state",
                )
            ]
            if data.get("motion_detection_zone_state")
            else None,
            motion_detection_score_std=data.get("motion_detection_score_std"),
            head_covering_type=head_covering_type,
            skeleton=SkeletonPb(**data["skeleton"])
            if data.get("skeleton")
            else None,
            ergonomic_severity_metrics=ErgonomicSeverityMetricsPb(
                **data["ergonomic_severity_metrics"]
            )
            if data.get("ergonomic_severity_metrics")
            else None,
        )

    def draw(self, img, label_type="pred"):
        # TODO: add text for more attributes.
        if self.polygon is None:
            return img

        top_left = self.polygon.get_top_left()
        bottom_right = self.polygon.get_bottom_right()

        box_color = None

        # TODO(harishma): Add actor filter logic here
        if self.category is ActorCategory.PERSON:
            box_color = (255, 0, 0)
        if self.category is ActorCategory.PIT:
            box_color = (0, 255, 0)
        if self.category is ActorCategory.SAFETY_VEST:
            box_color = (0, 255, 255)
        if self.category is ActorCategory.HARD_HAT:
            box_color = (0, 0, 255)
        if self.category is ActorCategory.DOOR:
            box_color = (255, 0, 255)

        if label_type == "gt":
            box_color = (255, 255, 255)
        text_color = [255, 255, 255]
        confidence = (
            round(self.confidence, 2) if self.confidence is not None else ""
        )
        track_id = self.track_id if self.track_id is not None else ""
        track_uuid = (
            self.track_uuid[-5:-1] if self.track_uuid is not None else ""
        )

        if self.category == ActorCategory.DOOR:
            if label_type == "gt":
                label = f"{self.door_state.name}"
            else:
                probabilities = [
                    str(round(prob, 2))
                    for prob in self.door_state_probabilities
                ]
                label = f"{self.door_state.name} : {' '.join(probabilities)}"
        else:
            label = f"{track_uuid if track_uuid else track_id} : {confidence}"

        if self.category is ActorCategory.PERSON:
            if self.activity:
                for activity_type in self.activity:
                    if (
                        activity_type == ActivityType.UNKNOWN.value
                        or self.activity[activity_type].activity
                        == ActivityType.UNKNOWN
                    ):
                        continue
                    label += f"""\nACT: {self.activity[activity_type].activity.name},
                     {self.activity[activity_type].posture.name}"""

                label += (
                    "\nCARRYING"
                    if self.is_carrying_object
                    else "\nNOT CARRYING"
                )

                label += (
                    "\nVEST" if self.is_wearing_safety_vest else "\nNO VEST"
                )
                label += "\nHAT" if self.is_wearing_hard_hat else "\nNO HAT"

        if box_color is not None:
            cv2.rectangle(
                img,
                (int(top_left.x), int(top_left.y)),
                (int(bottom_right.x), int(bottom_right.y)),
                box_color,
                2,
            )

            pos = (int(top_left.x), int(top_left.y))
            if label_type == "gt":
                bottom_left = self.polygon.get_bottom_left()
                # print gt labels on the bottom
                pos = (int(bottom_left.x), int(bottom_left.y))

            font_scale = min(img.shape[1], img.shape[0]) / (500 / 0.5)

            draw_text(
                img,
                label,
                pos=pos,
                font_scale=font_scale,
                text_color=text_color,
            )

        return img


class ActorFactory:
    """ActorFactory.

    This is a helper class to generate actors

    """

    def __init__(self):
        pass

    @classmethod
    def from_detection(
        cls,
        camera_uuid: str,
        track_id: int,
        bounding_box: list,
        category: ActorCategory,
        score: float,
        run_seed: Optional[str] = None,
    ) -> Actor:
        """from_detection

        Generates a Actor from a set of information generated from a track

        Args:
            camera_uuid (str): current camera uuid
            track_id (int): the track id from the tracker
            bounding_box (list): the TLWH bounding box from the detection
            category (ActorCategory): the actor category for the detection
            score (float): the detection score for the actor
            run_seed (str, Optional): optional run seed

        Returns:
            Actor: a new actor struct with the
        """
        actor = Actor()
        actor.uuid = camera_uuid
        actor.category = category
        actor.track_id = track_id
        actor.track_uuid = get_track_uuid(
            camera_uuid=camera_uuid,
            unique_identifier=str(track_id),
            category=category,
            run_seed=run_seed,
        )
        actor.polygon = Polygon.from_tlwh(bounding_box)
        actor.confidence = float(score)
        return actor
