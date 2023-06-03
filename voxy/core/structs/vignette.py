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

import attr

from core.structs.actor import ActorCategory
from core.structs.frame import Frame
from core.structs.tracklet import Tracklet

# trunk can't see these protos
# trunk-ignore-begin(pylint/E0611)
from protos.perception.structs.v1.vignette_pb2 import Vignette as VignettePb

# trunk-ignore-end(pylint/E0611)


@attr.s(slots=True)
class Vignette:
    """
    Represents the state of the system over a segment of time.
    """

    tracklets: typing.Dict[int, Tracklet] = attr.ib(factory=dict)
    past_frame_structs: typing.List[Frame] = attr.ib(factory=list)
    future_frame_structs: typing.List[Frame] = attr.ib(factory=list)
    present_frame_struct: typing.Optional[Frame] = attr.ib(default=None)
    present_timestamp_ms: typing.Optional[int] = attr.ib(default=None)

    def to_dict(self):
        """_summary_: Returns a dict representation of the Vignette instance.

        Returns:
            _type_: Dictionary
        """
        return {
            "tracklets": {
                tracklet_id: tracklet.to_dict()
                for tracklet_id, tracklet in self.tracklets.items()
            },
            "present_timestamp_ms": self.present_timestamp_ms
            if self.present_timestamp_ms is not None
            else None,
            "past_frame_structs": [
                frame_struct.to_dict()
                for frame_struct in self.past_frame_structs
            ],
            "future_frame_structs": [
                frame_struct.to_dict()
                for frame_struct in self.future_frame_structs
            ],
            "present_frame_struct": self.present_frame_struct.to_dict()
            if self.present_frame_struct is not None
            else None,
        }

    def get_timestamp_ns(self) -> int:
        """
        Returns the present timestamp in nanoseconds.

        Returns:
            int: the nanosecond timestamp
        """
        return (
            int(self.present_timestamp_ms * 1e6)
            if self.present_timestamp_ms is not None
            else 0
        )

    def to_proto(self) -> VignettePb:
        """
        Generates protobuf for vignette

        Returns:
            VignettePb: the vignette proto
        """
        return VignettePb(
            tracklets=list(
                tracklet.to_proto() for tracklet in self.tracklets.values()
            )
        )

    def to_annotation_protos(self) -> None:
        return None

    @classmethod
    def from_dict(cls, data: dict) -> object:
        """Creates a Vignette from a dictionary.

        Args:
            data: The data to construct the object from

        Returns:
            A Vignette object
        """
        return Vignette(
            tracklets={
                k: Tracklet.from_dict(v)
                for k, v in data.get("tracklets", {}).items()
            },
            present_timestamp_ms=data.get("present_timestamp_ms", None),
            present_frame_struct=Frame.from_dict(
                data.get("present_frame_struct", None)
            ),
            past_frame_structs=[
                Frame.from_dict(item)
                for item in data.get("past_frame_structs", [])
            ],
            future_frame_structs=[
                Frame.from_dict(item)
                for item in data.get("future_frame_structs", [])
            ],
        )

    def filter_null_xysr_tracks(
        self, actor_category: ActorCategory
    ) -> typing.Tuple[list, list]:
        """Filter null xysr tracks.

        Args:
            actor_category (ActorCategory): Actor Category to filter

        Returns:
            typing.Tuple[list, list]: A list of track ids and a list of xysr
        """
        track_id_actor = []
        xysr_actor = []

        for track_id, tracklet in self.tracklets.items():
            if tracklet.category == actor_category:
                xysr = tracklet.get_xysr_at_time(self.present_timestamp_ms)
                if xysr is not None:
                    track_id_actor.append(track_id)
                    xysr_actor.append(xysr)

        return track_id_actor, xysr_actor
