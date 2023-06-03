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

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Union

from cachetools import TTLCache
from loguru import logger

from core.structs.event import Event
from core.structs.incident import Incident
from core.structs.state import State

logging.getLogger("transitions").setLevel(logging.ERROR)


@dataclass
class RelevantActorInfo:
    actor_id: str
    track_uuid: str


class BaseStateMachine(ABC):

    NAME = None
    INCIDENT_TYPE_NAME = None
    INCIDENT_TYPE_ID = None
    PRE_START_BUFFER_MS = None
    POST_END_BUFFER_MS = None
    INCIDENT_VERSION = None
    INCIDENT_PRIORITY = None

    MAX_CACHE_SIZE = None
    MAX_CACHE_TIME_S = None

    # Base class variables
    PER_CAMERA_COOLDOWN_THRESHOLD_S = 0
    PER_CAMERA_COOLDOWN_THRESHOLD_MS = 0  # No cooldown by default

    PER_ACTOR_COOLDOWN_THRESHOLD_S = (
        24 * 3600
    )  # 24 hrs. This is essentially infinite cooldown
    PER_ACTOR_COOLDOWN_THRESHOLD_MS = 24 * 3600 * 1000

    def __init__(self, camera_uuid: str, params: dict) -> None:
        self.camera_uuid = camera_uuid
        self.params = params
        self._last_incident_timestamp_per_actor_ms = {}
        self._last_incident_timestamp_global_ms = None
        self.active_hours_start_utc = None
        self.active_hours_end_utc = None

        self._load_base_params(params)
        self.actor_machines = TTLCache(
            maxsize=self.MAX_CACHE_SIZE, ttl=self.MAX_CACHE_TIME_S
        )

    @abstractmethod
    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        """Virtual get_relevant_actor method to be implemented by base class, noop
        Args:
            state_event_message (Union[State, Event]): state or event message
        Returns:
            Optional[RelevantActorInfo]: relevant actor info to initialize actor machine
        Raises:
            RuntimeError: virutal method not implemented
        """
        raise RuntimeError(
            f"Virtual method get_relevant_actor must be implemented by {self.__class__}"
        )

    @abstractmethod
    def transition_actor_machine(
        self,
        actor_info: RelevantActorInfo,
        state_event_message: Union[State, Event],
    ) -> Optional[Incident]:
        pass

    def process_state_event(self, state_event_messages) -> list:
        if not self._check_class_variables():
            raise ValueError("All incident params must be set.")

        incident_list = []
        for state_event_message in state_event_messages:
            # do not process if timestamp is outside of active hours
            if not self._is_in_active_hours(state_event_message.timestamp_ms):
                continue

            actor_info = self.get_relevant_actor(state_event_message)

            if actor_info is None:
                continue

            generated_incident = self.transition_actor_machine(
                actor_info, state_event_message
            )

            if generated_incident is None:
                continue

            # add incident details
            generated_incident.title = self.INCIDENT_TYPE_NAME
            generated_incident.pre_start_buffer_ms = self.PRE_START_BUFFER_MS
            generated_incident.post_end_buffer_ms = self.POST_END_BUFFER_MS
            generated_incident.incident_type_id = self.INCIDENT_TYPE_ID
            generated_incident.incident_version = self.INCIDENT_VERSION

            # Set default priority if not set by ActorMachine
            if not generated_incident.priority:
                generated_incident.priority = self.INCIDENT_PRIORITY

            # Add to list of incidents if not cooling down
            violating_actor_ids = generated_incident.actor_ids
            if any(
                self._is_cooldown_period_for_actor(
                    actor_id, generated_incident.end_frame_relative_ms
                )
                for actor_id in violating_actor_ids
            ):
                generated_incident.cooldown_tag = True

            # Add cooldown tag if in per-camera cooldown
            if self._is_cooldown_period_for_camera(
                generated_incident.end_frame_relative_ms
            ):
                generated_incident.cooldown_tag = True

            self._last_incident_timestamp_global_ms = (
                generated_incident.end_frame_relative_ms
            )
            for actor_id in violating_actor_ids:
                self._last_incident_timestamp_per_actor_ms[
                    actor_id
                ] = generated_incident.end_frame_relative_ms

            incident_list.append(generated_incident)

        return incident_list

    def _is_cooldown_period_for_actor(self, actor_id, timestamp_ms):
        if actor_id not in self._last_incident_timestamp_per_actor_ms:
            return False
        return (
            self._last_incident_timestamp_per_actor_ms[actor_id]
            + self.PER_ACTOR_COOLDOWN_THRESHOLD_MS
            > timestamp_ms
        )

    def _is_cooldown_period_for_camera(self, timestamp_ms):
        if self._last_incident_timestamp_global_ms is None:
            return False
        return (
            self._last_incident_timestamp_global_ms
            + self.PER_CAMERA_COOLDOWN_THRESHOLD_MS
            > timestamp_ms
        )

    def _is_in_active_hours(self, timestamp_ms):
        """Returns if timestamp is in active hours

        Args:
            timestamp_ms (int): epoch timestamp in UTC

        Returns:
            bool: whether timestamp is in active hours
        """
        # if active hours are not set, all are valid
        if (
            self.active_hours_start_utc is None
            or self.active_hours_end_utc is None
        ):
            return True

        check_time = datetime.fromtimestamp(
            timestamp_ms / 1000.0, timezone.utc
        ).time()

        # if before midnight
        if self.active_hours_start_utc < self.active_hours_end_utc:
            return (
                self.active_hours_start_utc
                <= check_time
                <= self.active_hours_end_utc
            )

        # otherwise past midnight
        return (
            check_time >= self.active_hours_start_utc
            or check_time <= self.active_hours_end_utc
        )

    def _check_class_variables(self):
        def is_class_variable(x):
            return not x[0].startswith("_") and not inspect.ismethod(x[1])

        # explicitly define which class variables do not need to be set
        optional_class_variables = {
            "active_hours_end_utc",
            "active_hours_start_utc",
        }

        # check if all required class variables are set
        for class_variable in inspect.getmembers(self):
            if (
                is_class_variable(class_variable)
                and class_variable[0] not in optional_class_variables
                and class_variable[1] is None
            ):
                logger.warning(f"{class_variable[0]} is not set.")
                return False

        return True

    def _load_base_params(self, params):
        # Get params
        if params is not None:
            self.PER_CAMERA_COOLDOWN_THRESHOLD_S = params.get(
                "per_camera_cooldown_s", self.PER_CAMERA_COOLDOWN_THRESHOLD_S
            )
            self.PER_ACTOR_COOLDOWN_THRESHOLD_S = params.get(
                "per_actor_cooldown_s", self.PER_ACTOR_COOLDOWN_THRESHOLD_S
            )

            # Load timestamps
            active_hours_start_str = params.get("active_hours_start_utc", None)
            active_hours_end_str = params.get("active_hours_end_utc", None)

            if (
                active_hours_start_str is not None
                and active_hours_end_str is not None
            ):
                self.active_hours_start_utc = datetime.strptime(
                    active_hours_start_str, "%Y-%m-%dT%H:%M:%S.%fZ"
                ).time()
                self.active_hours_end_utc = datetime.strptime(
                    active_hours_end_str, "%Y-%m-%dT%H:%M:%S.%fZ"
                ).time()

        self.PER_ACTOR_COOLDOWN_THRESHOLD_MS = (
            1000 * self.PER_ACTOR_COOLDOWN_THRESHOLD_S
        )
        self.PER_CAMERA_COOLDOWN_THRESHOLD_MS = (
            1000 * self.PER_CAMERA_COOLDOWN_THRESHOLD_S
        )
