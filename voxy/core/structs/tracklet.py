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

# trunk-ignore(mypy/note,mypy/import)
import attr
import numpy as np
import scipy.interpolate
from cachetools import TTLCache
from sortedcontainers import SortedDict

from core.structs.actor import Actor, ActorCategory
from core.structs.attributes import Polygon

# trunk can't see these protos
# trunk-ignore-begin(pylint/E0611)
from protos.perception.structs.v1.actor_pb2 import (
    ActorCategory as ActorCategoryPb,
)
from protos.perception.structs.v1.tracklet_pb2 import Tracklet as TrackletPb

# trunk-ignore-end(pylint/E0611)

MAX_TRACKLET_ACTOR_INSTANCES = 1000


@attr.s(slots=True)
# trunk-ignore(pylint/R0902,pylint/R0904)
class Tracklet:
    """
    Tracklet class containing multiple instances of the same actor
    """

    expire_threshold_ms = attr.ib(type=int, default=10000)
    stationary_since_ms: typing.Optional[int] = attr.ib(default=None)
    is_believed_to_be_wearing_safety_vest: typing.Optional[bool] = attr.ib(
        default=None
    )
    is_believed_to_be_wearing_hard_hat: typing.Optional[bool] = attr.ib(
        default=None
    )
    is_believed_to_be_in_unsafe_posture: typing.Optional[bool] = attr.ib(
        default=None
    )
    is_associated_with_pit: typing.Optional[bool] = attr.ib(default=None)
    is_associated_with_person: typing.Optional[bool] = attr.ib(default=None)

    is_proximal_to_pit: typing.Optional[bool] = attr.ib(default=None)
    is_proximal_to_person: typing.Optional[bool] = attr.ib(default=None)

    nearest_pit_pixel_proximity = attr.ib(type=float, default=None)
    nearest_person_pixel_proximity = attr.ib(type=float, default=None)

    track_id = attr.ib(type=int, default=0)
    category: ActorCategory = attr.ib(default=ActorCategory.UNKNOWN)

    _instances = attr.ib(type=SortedDict, factory=SortedDict, init=False)
    # Note that initializing to 0 here means that a tracklet
    # with no instance will always return True when is_expired is called.
    _last_seen_timestamp_ms = attr.ib(type=int, default=0, init=False)

    _associations_map = attr.ib(type=dict, factory=dict, init=False)
    _proximity_map = attr.ib(type=dict, factory=dict, init=False)

    # Track the polygon of the actor at the given timestamp.
    _track = attr.ib(type=SortedDict, factory=SortedDict, init=False)
    timestamps = attr.ib(type=np.array, default=np.array([]))
    xysr_track = attr.ib(type=np.array, default=np.empty((4, 0)))
    is_stationary = attr.ib(type=bool, default=None)
    is_motion_zone_in_motion = attr.ib(type=bool, default=None)

    x_velocity_pixel_per_sec = attr.ib(type=float, default=None)
    y_velocity_pixel_per_sec = attr.ib(type=float, default=None)
    normalized_pixel_speed = attr.ib(type=float, default=None)
    # the window keeps track of all the normalized velocities in a window
    normalized_velocity_window = attr.ib(type=np.array, default=None)

    # Allows list style access to Tracklet
    def __getitem__(self, index: int) -> Actor:
        return self._instances.values()[index]

    def get_timestamps_and_actors(self):
        return zip(self._instances.keys(), self._instances.values())

    def get_actor_at_timestamp(
        self, timestamp_ms: int
    ) -> typing.Optional[Actor]:
        """get_actor_at_timestamp.

        Returns the actor instance at a given time. If the actor doesn't exist then
        it returns None

        Args:
            timestamp_ms (int): the time stamp in ms to query

        Returns:
            typing.Optional[Actor]: the optional actor at the time
        """
        return self._instances.get(timestamp_ms)

    def get_actors(self) -> typing.Iterable[Actor]:
        return self._instances.values()

    def get_last_seen_timestamp(self) -> int:
        return self._last_seen_timestamp_ms

    def __len__(self) -> int:
        return len(self._instances)

    def to_dict(self):
        """Returns a dict representation of the Tracklet instance."""
        return {
            "expire_threshold_ms": self.expire_threshold_ms,
            "stationary_since_ms": self.stationary_since_ms,
            "is_believed_to_be_wearing_hard_hat": self.is_believed_to_be_wearing_hard_hat,
            "is_believed_to_be_wearing_safety_vest": self.is_believed_to_be_wearing_safety_vest,
            "is_associated_with_pit": self.is_associated_with_pit,
            "is_associated_with_person": self.is_associated_with_person,
            "is_proximal_to_pit": self.is_proximal_to_pit,
            "is_proximal_to_person": self.is_proximal_to_person,
            "nearest_pit_pixel_proximity": self.nearest_pit_pixel_proximity,
            "nearest_person_pixel_proximity": self.nearest_person_pixel_proximity,
            "track_id": self.track_id,
            "category": self.category.name,
            "timestamps": self.timestamps.tolist(),
            "xysr_track": self.xysr_track.tolist(),
        }

    @classmethod
    def from_dict(cls, data):
        """_summary_: Creates a Tracklet from a dictionary."""
        return Tracklet(
            expire_threshold_ms=data.get("expire_threshold_ms"),
            stationary_since_ms=data.get("stationary_since_ms"),
            is_believed_to_be_wearing_hard_hat=data.get(
                "is_believed_to_be_wearing_hard_hat"
            ),
            is_believed_to_be_wearing_safety_vest=data.get(
                "is_believed_to_be_wearing_safety_vest"
            ),
            is_associated_with_pit=data.get("is_associated_with_pit"),
            is_associated_with_person=data.get("is_associated_with_person"),
            is_proximal_to_pit=data.get("is_proximal_to_pit"),
            is_proximal_to_person=data.get("is_proximal_to_person"),
            nearest_pit_pixel_proximity=data.get(
                "nearest_pit_pixel_proximity"
            ),
            nearest_person_pixel_proximity=data.get(
                "nearest_person_pixel_proximity"
            ),
            track_id=data.get("track_id"),
            category=ActorCategory[data.get("category")],
            timestamps=np.array(data.get("timestamps")),
            xysr_track=np.array(data.get("xysr_track")),
        )

    def to_proto(self) -> TrackletPb:
        """
        Generates lightweight protobuf for tracklet

        Returns:
            TrackletPb: the tracklet
        """

        return TrackletPb(
            expire_threshold_ms=self.expire_threshold_ms,
            stationary_since_ms=self.stationary_since_ms,
            is_believed_to_be_wearing_hard_hat=self.is_believed_to_be_wearing_hard_hat,
            is_believed_to_be_wearing_safety_vest=self.is_believed_to_be_wearing_safety_vest,
            is_associated_with_pit=self.is_associated_with_pit,
            is_associated_with_person=self.is_associated_with_person,
            track_id=self.track_id,
            timestamps=self.timestamps.tolist(),
            category=ActorCategoryPb.Value(
                f"ACTOR_CATEGORY_{self.category.name}"
            ),
            is_believed_to_be_in_unsafe_posture=self.is_believed_to_be_in_unsafe_posture,
            is_stationary=self.is_stationary,
            is_motion_zone_in_motion=self.is_motion_zone_in_motion,
            x_velocity_pixel_per_sec=self.x_velocity_pixel_per_sec,
            y_velocity_pixel_per_sec=self.y_velocity_pixel_per_sec,
            normalized_pixel_speed=self.normalized_pixel_speed,
        )

    def to_annotation_protos(self):
        return None

    def xysr(self, i: int) -> np.array:
        return self.xysr_track[:, i]

    def is_expired_at(self, timestamp_ms: int) -> bool:
        # TODO(harishma): Raise if timestamp_ms is negative
        return (
            len(self._instances) == 0
            or timestamp_ms - self._last_seen_timestamp_ms
            > self.expire_threshold_ms
        )

    def earliest_available_timestamp_ms(self) -> typing.Optional[int]:
        return self._instances.keys()[0] if len(self._instances) > 0 else None

    def update(self, instance: Actor, timestamp_ms: int) -> None:
        # TODO: consider checking track id is the same
        # Limit the number of actor instances that are stored.
        if len(self._instances) == MAX_TRACKLET_ACTOR_INSTANCES:
            removed_timestamp, removed_item = self._instances.popitem(index=0)
            if removed_item is not None and len(self._track) > 0:
                self._track.popitem(index=0)

        self._instances[timestamp_ms] = instance

        if instance is not None:
            self._last_seen_timestamp_ms = timestamp_ms
            self.category = instance.category
            self.track_id = instance.track_id

            self._track[timestamp_ms] = self._convert_bbox_to_xysr(
                instance.polygon
            )
            # find the timestamp we want
            self.timestamps = np.append(
                self.timestamps, [timestamp_ms]
            ).astype(np.int64)
            index = np.where(self.timestamps == self._track.keys()[0])[0][0]
            self.timestamps = self.timestamps[index:]
            # append to the timestamp array
            if self.category in [
                ActorCategory.PIT,
                ActorCategory.PERSON,
                ActorCategory.OBSTRUCTION,
            ]:
                new_xysr = self._track[timestamp_ms]
                self.xysr_track = np.append(self.xysr_track, new_xysr, axis=1)
                # pop old indexes
                self.xysr_track = self.xysr_track[:, index:]

    # Return instances with timestamp greater than or equal to start_timestamp_ms and less than or equal to end_timestamp_ms.
    def get_actor_instances_in_time_interval(
        self, start_timestamp_ms: int, end_timestamp_ms: int
    ) -> map:
        return map(
            self._instances.__getitem__,
            self._instances.irange(
                start_timestamp_ms, end_timestamp_ms, inclusive=(True, True)
            ),
        )

    # If allow_closest is True and no actor instance exists at the given timestamp,
    # the actor index with closest timestamp but strictly lesser than the given timestamp
    # is returned.
    def get_actor_index_at_time(
        self, timestamp_ms: int, allow_closest_earlier_timestamp: bool = False
    ) -> typing.Union[int, None]:
        if timestamp_ms in self._instances:
            return self._instances.index(timestamp_ms)

        index = (
            self._instances.bisect_left(timestamp_ms) - 1
            if allow_closest_earlier_timestamp
            else None
        )

        return index if index is not None and index >= 0 else None

    # Returns the index of the closest non Null/None actor instance
    def get_closest_non_null_actor_at_time(
        self, timestamp_ms: int
    ) -> typing.Optional[int]:
        if len(self._instances) == 0:
            return None

        timestamps = np.array(self._instances.keys()[:])
        sort_list = np.argsort(np.abs(timestamps - timestamp_ms))

        for i in sort_list:
            if self._instances[timestamps[i]] is not None:
                return i
        return None

    # Returns the bounding box at a time
    # TODO(Vai): test
    def get_bbox_at_time(
        self, timestamp_ms: int, interpolate: bool = False
    ) -> typing.Optional[typing.List[float]]:
        return self._convert_xysr_to_bbox(
            self.get_xysr_at_time(timestamp_ms, interpolate)
        )

    # Returns xysr bounding box at a time
    # IMPORTANT: time range assumes interpolation
    # TODO(Vai): test
    def get_bbox_at_time_range(
        self, timestamp_range_ms: typing.List[int]
    ) -> typing.Optional[typing.List[float]]:
        # TODO(Vai): do time range calculation
        return self.get_xysr_at_time_range(timestamp_range_ms)

    def get_closest_timestamp_idx_list(self, timestamp: int) -> np.array:
        return np.argsort(np.abs(self.timestamps - timestamp))

    # Returns xysr bounding box at a time
    def get_xysr_at_time(
        self, timestamp_ms: int, interpolate: bool = False
    ) -> typing.Optional[typing.List[float]]:

        if not interpolate:
            sort_list = self.get_closest_timestamp_idx_list(timestamp_ms)

            if sort_list.shape[0]:
                return self.xysr_track[:, sort_list[0]]

            return None

        # TODO(vai): Cleanup the indexing and ranges
        INTERPOLATION_WINDOW_SIZE_S = 3

        min_ind = self.get_closest_timestamp_idx_list(
            timestamp_ms - INTERPOLATION_WINDOW_SIZE_S * 1000
        )[0]
        max_ind = self.get_closest_timestamp_idx_list(
            timestamp_ms + INTERPOLATION_WINDOW_SIZE_S * 1000
        )[0]

        timestamps = self.timestamps[min_ind : max_ind + 1]
        xysr = self.xysr_track[:, min_ind : max_ind + 1]

        if len(timestamps) <= 1:
            return xysr[:, 0] if len(timestamps) > 0 else None

        x_array = xysr[0, :]
        y_array = xysr[1, :]
        s_array = xysr[2, :]
        r_array = xysr[3, :]

        # Set interpolation level
        interpolation_type = "nearest"
        if len(timestamps) == 2:
            interpolation_type = "linear"
        elif len(timestamps) == 3:
            interpolation_type = "quadratic"
        elif len(timestamps) >= 4:
            interpolation_type = "cubic"

        # TODO(Vai): Merge into a single interpolate
        x_interp = scipy.interpolate.interp1d(
            timestamps,
            x_array,
            kind=interpolation_type,
            bounds_error=False,
            fill_value=(x_array[0], x_array[-1]),
            assume_sorted=True,
        )
        y_interp = scipy.interpolate.interp1d(
            timestamps,
            y_array,
            kind=interpolation_type,
            bounds_error=False,
            fill_value=(y_array[0], y_array[-1]),
            assume_sorted=True,
        )
        s_interp = scipy.interpolate.interp1d(
            timestamps,
            s_array,
            kind=interpolation_type,
            bounds_error=False,
            fill_value=(s_array[0], s_array[-1]),
            assume_sorted=True,
        )
        r_interp = scipy.interpolate.interp1d(
            timestamps,
            r_array,
            kind=interpolation_type,
            bounds_error=False,
            fill_value=(r_array[0], r_array[-1]),
            assume_sorted=True,
        )

        # interpolate
        return np.array(
            [
                x_interp(timestamp_ms),
                y_interp(timestamp_ms),
                s_interp(timestamp_ms),
                r_interp(timestamp_ms),
            ]
        )

    # Returns xysr bounding box at a time
    # IMPORTANT: time range assumes interpolation
    # TODO(Vai): test
    def get_xysr_at_time_range(
        self, timestamp_range_ms: typing.List[int]
    ) -> typing.Optional[typing.List[float]]:

        # TODO(vai): Cleanup the indexing and ranges
        INTERPOLATION_WINDOW_SIZE_S = 3

        min_ind = self.get_closest_timestamp_idx_list(
            timestamp_range_ms[0] - INTERPOLATION_WINDOW_SIZE_S * 1000
        )[0]
        max_ind = self.get_closest_timestamp_idx_list(
            timestamp_range_ms[-1] + INTERPOLATION_WINDOW_SIZE_S * 1000
        )[0]

        timestamps = self.timestamps[min_ind : max_ind + 1]
        xysr = self.xysr_track[:, min_ind : max_ind + 1]

        if len(timestamps) <= 1:
            return np.array(xysr[:, 0]) if len(timestamps) > 0 else None

        # Set interpolation level
        interpolation_type = "linear"

        x_array = xysr[0, :]
        y_array = xysr[1, :]
        s_array = xysr[2, :]
        r_array = xysr[3, :]

        x_interp = scipy.interpolate.interp1d(
            timestamps,
            x_array,
            kind=interpolation_type,
            bounds_error=False,
            fill_value=(x_array[0], x_array[-1]),
            assume_sorted=True,
        )
        y_interp = scipy.interpolate.interp1d(
            timestamps,
            y_array,
            kind=interpolation_type,
            bounds_error=False,
            fill_value=(y_array[0], y_array[-1]),
            assume_sorted=True,
        )
        s_interp = scipy.interpolate.interp1d(
            timestamps,
            s_array,
            kind=interpolation_type,
            bounds_error=False,
            fill_value=(s_array[0], s_array[-1]),
            assume_sorted=True,
        )
        r_interp = scipy.interpolate.interp1d(
            timestamps,
            r_array,
            kind=interpolation_type,
            bounds_error=False,
            fill_value=(r_array[0], r_array[-1]),
            assume_sorted=True,
        )

        # interpolate
        return np.vstack(
            [
                x_interp(timestamp_range_ms),
                y_interp(timestamp_range_ms),
                s_interp(timestamp_range_ms),
                r_interp(timestamp_range_ms),
            ]
        ).transpose()

    # TODO move below functions to core/structs/attributes
    # Copied from core/perception/detector_tracker/sort.py
    def _convert_bbox_to_xysr(
        self, polygon: Polygon
    ) -> typing.Optional[np.array]:
        """Convert bbox to z.

        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        if polygon is None:
            return None

        w = polygon.get_bottom_right().x - polygon.get_top_left().x
        h = polygon.get_bottom_right().y - polygon.get_top_left().y
        x = polygon.get_top_left().x + w / 2.0
        y = polygon.get_top_left().y + h / 2.0
        s = w * h  # scale is just area
        r = h / float(w)

        return np.array([x, y, s, r]).reshape((4, 1))

    # Copied from core/perception/detector_tracker/sort.py
    def _convert_xysr_to_bbox(
        self, xysr: np.array
    ) -> typing.Optional[np.array]:
        """Convert z to bbox.

        Takes a bounding box in the form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        if xysr is None:
            return None

        x, y, s, r = xysr[0], xysr[1], xysr[2], xysr[3]
        w = np.sqrt(s / r)
        h = r * w
        x1 = x - w / 2.0
        x2 = x + w / 2.0
        y1 = y - h / 2.0
        y2 = y + h / 2.0

        return np.array([x1, y1, x2, y2]).transpose()

    def get_timestamp_at_index(self, index: int) -> typing.Union[int, None]:
        # Returns timestamp at a given index
        # If index doesn't exists returns None
        keys = self._instances.keys()
        return keys[index] if -1 * len(keys) <= index < len(keys) else None

    def update_tracklet_associations(
        self,
        actor_category: ActorCategory,
        frame_timestamp_ms: int,
        associated_track_id: int,
    ) -> None:
        self._associations_map = self._associations_map or {}
        if actor_category not in self._associations_map:
            self._associations_map[actor_category] = SortedDict()
        temporal_associations: SortedDict = self._associations_map[
            actor_category
        ]
        if len(temporal_associations) == MAX_TRACKLET_ACTOR_INSTANCES:
            temporal_associations.popitem(index=0)
        temporal_associations[frame_timestamp_ms] = associated_track_id

    def update_tracklet_proximity(
        self,
        proximal_actors_dict: typing.Dict[int, float],
        frame_timestamp_ms: int,
        proximal_actors_category: ActorCategory,
    ) -> None:
        """Update the tracklet proximity map with a dictionary of proximal actors
           and their corresponding distances

        Args:
            proximal_actors_dict (dict): All actors proximal to tracklet actor and distances
            frame_timestamp_ms (int): Vignette present timestamps
            proximal_actors_category (ActorCategory): Actor category of proximal actors
        """
        if proximal_actors_category not in self._proximity_map:
            self._proximity_map[proximal_actors_category] = TTLCache(
                maxsize=50, ttl=10
            )
        self._proximity_map[proximal_actors_category][
            frame_timestamp_ms
        ] = proximal_actors_dict

    def _get_closest_association_timestamp(
        self, actor_category: ActorCategory, timestamp_ms: int
    ) -> int:
        self._associations_map = self._associations_map or {}
        if actor_category not in self._associations_map:
            return None
        temporal_associations: SortedDict = self._associations_map[
            actor_category
        ]
        if not len(temporal_associations):
            return None
        timestamps_ms = np.array(temporal_associations.keys()[:])
        closest_timestamp_idx = np.argsort(
            np.abs(timestamps_ms - timestamp_ms)
        )[0]
        return timestamps_ms[closest_timestamp_idx]

    def get_raw_associated_tracklet_at_time(
        self,
        actor_category: ActorCategory,
        frame_timestamp_ms: int,
    ) -> int:
        self._associations_map = self._associations_map or {}
        if actor_category not in self._associations_map:
            return None
        temporal_associaions: SortedDict = self._associations_map[
            actor_category
        ]
        if not len(temporal_associaions):
            return None
        closest_time = self._get_closest_association_timestamp(
            actor_category, frame_timestamp_ms
        )
        return temporal_associaions[closest_time]

    def get_temporal_association_for_actor(
        self, actor_category: ActorCategory
    ) -> SortedDict:
        self._associations_map = self._associations_map or {}

        return (
            self._associations_map[actor_category]
            if actor_category in self._associations_map
            else None
        )

    # TODO (Gabriel): Add future for smoothing - requires rearch
    def get_smoothed_association_for_actor(
        self,
        actor_category: ActorCategory,
        timestamp_ms: int,
        smoothing_lookback: int,
        probability_to_associate: float,
    ) -> int:
        self._associations_map = self._associations_map or {}
        if actor_category not in self._associations_map:
            return None
        temporal_associations = self._associations_map[actor_category]
        if not len(temporal_associations):
            return None
        earliest_timestamp_ms = timestamp_ms - smoothing_lookback
        associations_in_range = list(
            map(
                temporal_associations.__getitem__,
                temporal_associations.irange(
                    earliest_timestamp_ms, timestamp_ms, inclusive=(True, True)
                ),
            )
        )
        most_associated = max(
            set(associations_in_range), key=associations_in_range.count
        )
        frequency = float(
            associations_in_range.count(most_associated)
            / len(associations_in_range)
        )
        return (
            most_associated if frequency > probability_to_associate else None
        )

    def clear_temporal_association_for_actor(
        self, actor_category: ActorCategory
    ) -> None:
        self._associations_map = self._associations_map or {}
        if actor_category in self._associations_map:
            self._associations_map.pop(actor_category)
