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
import attr
from mcap_protobuf.writer import Writer as ProtoWriter

from .frame import Frame


@attr.s(slots=True)
class Video:

    uuid = attr.ib(type=str)
    parent_uuid = attr.ib(default=None, type=str)
    root_uuid = attr.ib(default=None, type=str)
    name = attr.ib(default=None, type=str)
    path = attr.ib(default=None, type=str)
    voxel_uuid = attr.ib(default=None, type=str)
    is_test = attr.ib(default=None, type=bool)

    # TODO add validators to ensure items are type of Frame.
    frames = attr.ib(factory=list)

    def to_dict(self):
        return {
            "uuid": self.uuid,
            "parent_uuid": self.parent_uuid,
            "root_uuid": self.root_uuid,
            "frames": [frame.to_dict() for frame in self.frames],
            "voxel_uuid": self.voxel_uuid,
            "is_test": self.is_test,
        }

    @classmethod
    def from_dict(cls, data):
        return Video(
            uuid=data["uuid"],
            parent_uuid=data["parent_uuid"],
            root_uuid=data["root_uuid"],
            voxel_uuid=data.get("voxel_uuid"),
            is_test=data.get("is_test"),
            frames=[Frame.from_dict(frame) for frame in data["frames"]],
        )

    @classmethod
    def from_metaverse(cls, data):
        return Video(
            uuid=data["uuid"],
            name=data["name"],
            path=data["path"],
            voxel_uuid=data["voxel_uuid"],
            is_test=data["is_test"],
            frames=[
                Frame.from_metaverse(frame) for frame in data["frame_ref"]
            ],
        )

    def serialize_to_mcap(self, filename: str):
        """
        Serializes the Video struct to the MCAP log file

        Args:
            filename (str): the filename of the log file to write to
        """
        with ProtoWriter(filename) as writer:
            for frame in self.frames:
                writer.write_message(
                    topic="PerceptionNode",
                    message=frame.to_proto(),
                    log_time=frame.get_timestamp_ns(),
                    publish_time=frame.get_timestamp_ns(),
                )
