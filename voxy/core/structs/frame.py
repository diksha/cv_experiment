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
from dataclasses import dataclass
from enum import Enum

import attr
import cv2
import numpy as np
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.ImageAnnotations_pb2 import ImageAnnotations
from google.protobuf.timestamp_pb2 import Timestamp

from core.common.utils.proto_utils import VoxelProto
from core.structs.actor import Actor

# trunk-ignore-begin(pylint/E0611)
from protos.perception.structs.v1.frame_pb2 import Frame as FramePb
from protos.perception.types.v1.types_pb2 import Timestamp as TimestampPb

# trunk-ignore-end(pylint/E0611)


class FrameSegmentCategory(Enum):
    """All available class types for frame segments"""

    UNKNOWN = 0
    FLOOR = 1
    SPILL = 2
    OBSTRUCTION = 3


@dataclass
class StampedImage:
    image: np.ndarray
    timestamp_ms: int

    def get_timestamp_ns(self) -> int:
        """
        Get's the current nanosecond timestamp

        Returns:
            int: current timestamp of the frame in nanoseconds
        """
        return int(self.timestamp_ms * 1e6)

    def to_proto(self) -> CompressedImage:
        """
        Serializes the current image to a protobuf

        Returns:
            CompressedImage: the current image serialized
        """
        image = self.image
        timestamp_ms = self.timestamp_ms
        image_bytes = bytes(cv2.imencode(".jpg", image)[1].tostring())
        timestamp = Timestamp()
        timestamp.FromMilliseconds(millis=timestamp_ms)
        compressed_image = CompressedImage(
            data=image_bytes,
            format="jpeg",
            timestamp=timestamp,
        )
        return compressed_image

    def get_topic_name(self) -> str:
        """
        Get's the intended topic name of the image

        Returns:
            str: the topic name "image_frame"
        """
        return "image_frame"


@attr.s(slots=True)
# trunk-ignore(pylint/R0902)
class Frame:
    """Represents a single video frame."""

    frame_number = attr.ib(type=int)

    frame_width = attr.ib(type=int)
    frame_height = attr.ib(type=int)

    # Should we create a timestamp class to store multiple timestamps and/or
    # validate?

    # TODO: Remove timestamp in seconds and
    # populate relative and epoch correctly
    # for both video and stream.
    # For video use epoch as current time when video
    # starts processing and add relative time to it.
    relative_timestamp_s = attr.ib(type=float)
    relative_timestamp_ms = attr.ib(type=float)
    epoch_timestamp_ms = attr.ib(type=float)

    # We make the assumption that only one segmentation label per pixel is allowed
    frame_segments = attr.ib(type=np.ndarray, default=None)

    # TODO add validators to ensure items are type of Actor.
    actors = attr.ib(factory=list)
    # Relative path of the frame to the parent data collector or video.
    relative_image_path = attr.ib(default=None, type=str)

    def to_dict(self) -> dict:
        """
        Serializes a Frame instance to json

        Returns:
            A dict representation of the frame instance.
        """
        return {
            "frame_number": self.frame_number,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "relative_timestamp_s": self.relative_timestamp_s,
            "relative_timestamp_ms": self.relative_timestamp_ms,
            "epoch_timestamp_ms": self.epoch_timestamp_ms,
            "frame_segments": self.frame_segments.tolist()
            if self.frame_segments is not None
            else None,
            "actors": [actor.to_dict() for actor in self.actors],
            "relative_image_path": self.relative_image_path,
        }

    def get_timestamp_ns(self) -> int:
        """
        Generates the epoch timestamp in milliseconds

        Returns:
            int: the current timestamp in nanoseconds
        """
        return int(self.epoch_timestamp_ms * 1e6)

    @classmethod
    def from_dict(cls, data: dict) -> object:
        """
        Generates a Frame instance from a dict

        Args:
            data: a dictionary containing a Frame instance as json

        Returns:
            An instance of Frame class
        """
        return Frame(
            frame_number=data.get("frame_number", None),
            frame_width=data.get("frame_width", None),
            frame_height=data.get("frame_height", None),
            relative_timestamp_s=data.get("relative_timestamp_s", None),
            relative_timestamp_ms=data.get("relative_timestamp_ms", None),
            epoch_timestamp_ms=data.get("epoch_timestamp_ms", None),
            frame_segments=np.array(data["frame_segments"])
            if "frame_segments" in data and data["frame_segments"] is not None
            else None,
            actors=[
                Actor.from_dict(actor) for actor in data.get("actors", [])
            ],
            relative_image_path=data.get("relative_image_path", None),
        )

    @classmethod
    def from_metaverse(cls, data):
        """Get frame object from metaverse

        Args:
            data (_type_):input data containing actor info

        Returns:
            _type_:output frame object
        """
        actors = []
        for actor in data.get("actors_ref"):
            actors.append(Actor.from_metaverse(actor))
        return Frame(
            frame_number=data.get("frame_number", None),
            frame_width=data.get("frame_width", None),
            frame_height=data.get("frame_height", None),
            relative_timestamp_s=data.get("relative_timestamp_s", None),
            relative_timestamp_ms=data.get("relative_timestamp_ms", None),
            epoch_timestamp_ms=data.get("epoch_timestamp_ms", None),
            actors=actors,
            relative_image_path=data.get("relative_image_path", None),
        )

    def to_proto(self):
        """Converts frame python class to frame.proto

        Returns:
            Frame: a frame.proto
        """
        # create frame proto
        frame_pb = FramePb(
            frame_number=self.frame_number,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            timestamp=TimestampPb(
                relative_timestamp_ms=self.relative_timestamp_ms,
                epoch_timestamp_ms=self.epoch_timestamp_ms,
            ),
            relative_image_path=self.relative_image_path,
        )

        # create array of actors
        frame_pb.actors.extend([actor.to_proto() for actor in self.actors])

        return frame_pb

    @classmethod
    def from_proto(cls, raw_proto: FramePb) -> "Frame":
        """
        Generates a protobuf from a frame protobuf

        Args:
            raw_proto (Any): the raw proto to convert from

        Returns:
            Frame: the frame generated from the frame proto
        """
        proto = VoxelProto(raw_proto)

        actors = [Actor.from_proto(actor) for actor in raw_proto.actors]
        return Frame(
            frame_number=proto.frame_number,
            frame_width=proto.frame_width,
            frame_height=proto.frame_height,
            relative_timestamp_ms=proto.timestamp.relative_timestamp_ms
            if proto.timestamp is not None
            else None,
            relative_timestamp_s=proto.timestamp.relative_timestamp_ms / 1000
            if proto.timestamp is not None
            and proto.timestamp.relative_timestamp_ms is not None
            else None,
            epoch_timestamp_ms=proto.timestamp.epoch_timestamp_ms
            if proto.timestamp
            else None,
            relative_image_path=proto.relative_image_path,
            actors=actors,
        )

    def to_annotation_protos(self) -> ImageAnnotations:
        """
        Returns the annotations as an ImageAnnotations
        object

        Returns:
            ImageAnnotations: the image annotations
        """
        # all annotations are flat messages
        annotations = [actor.to_annotation_protos() for actor in self.actors]
        text = [actor.to_text_annotation_protos() for actor in self.actors]
        # add more info to the frame maybe?
        pose_annotations = [
            actor.to_pose_annotation_protos()
            for actor in self.actors
            if actor.has_pose()
        ]

        image_annotation = ImageAnnotations(
            points=annotations,
            texts=text,
        )
        for pose_annotation in pose_annotations:
            image_annotation.points.extend(pose_annotation.points)
            image_annotation.texts.extend(pose_annotation.texts)
            image_annotation.circles.extend(pose_annotation.circles)
        return image_annotation

    def draw(
        self,
        img: np.ndarray,
        label_type="pred",
        categories_to_keep=None,
        actionable_region=None,
        actor_ids=None,
        draw_timestamp=True,
    ) -> np.ndarray:
        """Draws each actor described by this frame onto an image.

        Args:
            img: Frame image to draw on.
            label_type (string): Type of labels to draw.
            categories_to_keep: Categories that will be drawn. All not-None categories are drawn.
            actionable_region: Points of actionable region
            actor_ids: List of actor track ids, if provided only those will be drawn.
            draw_timestamp (bool): True if frame timestamp label should be drawn, otherwise False.

        Returns:
            np.ndarray: output image
        """
        if draw_timestamp:
            cv2.putText(
                img,
                str(self.relative_timestamp_ms),
                (10, 100),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                [255, 255, 255],
                2,
            )
        for actor in self.actors:
            if actor_ids is not None:
                draw_actor = (
                    actor.track_id in actor_ids
                    or str(actor.track_id) in actor_ids
                )
                if not draw_actor:
                    continue

            if (
                categories_to_keep is None
                or actor.category.name in categories_to_keep
            ):
                img = actor.draw(img, label_type=label_type)

        if actionable_region is not None:
            cv2.polylines(
                img,
                actionable_region,
                isClosed=True,
                color=(150, 0, 150),
                thickness=3,
            )
        return img
