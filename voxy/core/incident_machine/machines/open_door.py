#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

from typing import Optional, Union

from transitions import Machine

from core.incident_machine.machines.base import (
    BaseStateMachine,
    RelevantActorInfo,
)
from core.structs.actor import ActorCategory
from core.structs.actor import DoorState as DoorStateEnum
from core.structs.event import Event
from core.structs.incident import Incident
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class ActorMachine:

    MAX_OPEN_DOOR_S = 60
    MAX_OPEN_DOOR_MS = MAX_OPEN_DOOR_S * 1000

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ) -> None:
        self.camera_uuid = camera_uuid
        self.actor_info = actor_info

        if params is not None:
            self.MAX_OPEN_DOOR_S = params.get(
                "max_open_door_s", self.MAX_OPEN_DOOR_S
            )
            self.MAX_OPEN_DOOR_MS = self.MAX_OPEN_DOOR_S * 1000

        self.door_opened_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None

        self._is_in_cooldown = False
        self.sequence_id = 0

        self._initialize_state_machine()

    def _initialize_state_machine(self) -> None:
        states = ["closed", "partially_open", "open", "incident"]

        transitions = [
            {
                "trigger": "update_door_state",
                "source": "*",
                "dest": "closed",
                "conditions": "_is_door_state_closed",
            },
            {
                "trigger": "update_door_state",
                "source": "open",
                "dest": None,
                "conditions": "_is_door_state_open",
            },
            {
                "trigger": "update_door_state",
                "source": ["closed", "partially_open"],
                "dest": "open",
                "conditions": "_is_door_state_open",
            },
            {
                "trigger": "update_door_state",
                "source": "partially_open",
                "dest": None,
                "conditions": "_is_door_state_partially_open",
            },
            {
                "trigger": "update_door_state",
                "source": ["closed", "open"],
                "dest": "partially_open",
                "conditions": "_is_door_state_partially_open",
            },
            {
                "trigger": "update_door_state",
                "source": "incident",
                "dest": None,
            },
            {
                "trigger": "check_incident",
                "source": ["partially_open", "open"],
                "conditions": [
                    "_is_door_open_for_greater_than_time_threshold",
                ],
                "dest": "incident",
                "after": "_get_incident",
            },
            {
                "trigger": "check_incident",
                "source": "*",
                "dest": None,
            },
            {
                "trigger": "try_reset_state",
                "source": "incident",
                "dest": "partially_open",
                "conditions": "_is_door_state_partially_open",
            },
            {
                "trigger": "try_reset_state",
                "source": "incident",
                "dest": "open",
                "conditions": "_is_door_state_open",
            },
            {
                "trigger": "try_reset_state",
                "source": [
                    "closed",
                    "partially_open",
                    "open",
                ],
                "dest": None,
            },
        ]
        Machine(
            model=self,
            states=states,
            transitions=transitions,
            initial="closed",
            ignore_invalid_triggers=False,
        )

    def transition_state_event(
        self, state_event_message: Union[State, Event]
    ) -> Optional[Incident]:

        incident_list = []

        self.state_event_start_timestamp_ms = state_event_message.timestamp_ms
        self.state_event_end_timestamp_ms = (
            state_event_message.end_timestamp_ms
        )

        self.update_door_state(
            door_state=state_event_message.door_state,
        )

        self.check_incident(incident_list=incident_list)

        self.try_reset_state(door_state=state_event_message.door_state)

        return incident_list[0] if len(incident_list) > 0 else None

    # Conditionals
    def _is_door_open_for_greater_than_time_threshold(self, **kwargs) -> bool:
        """Checks the condition for generating incident when door_state is partially open or open

        Returns:
            bool: True if an incident occurs (time>threshold) else False
        """
        return (
            self.state_event_end_timestamp_ms - self.door_opened_timestamp_ms
            > self.MAX_OPEN_DOOR_MS
        )

    def _is_door_state_partially_open(self, door_state, **_) -> bool:
        """Now we consider partially open state for incidents
        Args:
            door_state (_type_): state of the door (Enum)

        Returns:
            bool: True if door is partially open
        """
        return door_state == DoorStateEnum.PARTIALLY_OPEN

    def _is_door_state_open(self, door_state, **kwargs) -> bool:
        return door_state == DoorStateEnum.FULLY_OPEN

    def _is_door_state_closed(self, door_state, **kwargs) -> bool:
        return door_state == DoorStateEnum.FULLY_CLOSED

    # Callbacks
    def _get_incident(self, incident_list, **kwargs) -> None:
        incident_list.append(
            Incident(
                camera_uuid=self.camera_uuid,
                start_frame_relative_ms=self.door_opened_timestamp_ms,
                end_frame_relative_ms=self.state_event_end_timestamp_ms,
                actor_ids=[self.actor_info.actor_id],
                track_uuid=self.actor_info.track_uuid,
                cooldown_tag=self._is_in_cooldown,
                sequence_id=self.sequence_id,
            )
        )
        # Any incident raised after first one is considered cooldown incident unless the
        # state machine is fully reset.
        self._is_in_cooldown = True

    def on_enter_open(self, **kwargs) -> None:
        self.door_opened_timestamp_ms = self.state_event_start_timestamp_ms

    def on_enter_partially_open(self, **_) -> None:
        """Restart the door_partially_opened timestamp to the start state timestamp
        when we transition to the partially open state
        Args:
            **_ : kwargs
        """
        self.door_opened_timestamp_ms = self.state_event_start_timestamp_ms

    def on_enter_closed(self, **kwargs) -> None:
        """Reset tracking variables upon entering closed state
        Args:
            **kwargs : kwargs
        """
        self.door_opened_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None
        self._is_in_cooldown = False
        self.sequence_id += 1


class OpenDoorMachine(BaseStateMachine):
    """TODO(harishma): Add a comment documenting the caveats with this machine and how it differs
    from all other incident machines which deal with dynamic actors.
    """

    # State Machine Diagram
    # https://drive.google.com/file/d/1uCeabHDhS8tR7nc3b2iSFPBOYveBl6q9/view?usp=share_link

    NAME = "open_door"
    INCIDENT_TYPE_NAME = "Open Door duration"
    INCIDENT_TYPE_ID = "OPEN_DOOR_DURATION"
    PRE_START_BUFFER_MS = 6000
    POST_END_BUFFER_MS = 6000
    INCIDENT_VERSION = "1.0"
    INCIDENT_PRIORITY = "medium"

    # This machine by default should not suppress incidents by actor id
    # as the actor is door. Because it's a static actor the id never changes.
    PER_ACTOR_COOLDOWN_THRESHOLD_S = 0
    PER_ACTOR_COOLDOWN_THRESHOLD_MS = PER_ACTOR_COOLDOWN_THRESHOLD_S * 1000

    # Cache Configs
    MAX_CACHE_SIZE = 5  # max 5 actors in a scene
    MAX_CACHE_TIME_S = 12 * 60 * 60  # 12 hours

    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        if (
            isinstance(state_event_message, State)
            and state_event_message.actor_category == ActorCategory.DOOR
        ):
            return RelevantActorInfo(
                actor_id=state_event_message.actor_id,
                track_uuid=state_event_message.track_uuid,
            )

        return None

    def transition_actor_machine(
        self,
        actor_info: RelevantActorInfo,
        state_event_message: Union[State, Event],
    ) -> Optional[Incident]:

        if actor_info.track_uuid not in self.actor_machines:
            # create the machine first
            self.actor_machines[actor_info.track_uuid] = ActorMachine(
                self.camera_uuid,
                actor_info,
                params=self.params,
            )

        actor_machine = self.actor_machines[actor_info.track_uuid]
        return actor_machine.transition_state_event(state_event_message)
