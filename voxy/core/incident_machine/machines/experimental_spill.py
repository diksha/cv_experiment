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

from typing import List, Optional, Union

from transitions import Machine

from core.incident_machine.machines.base import (
    BaseStateMachine,
    RelevantActorInfo,
)
from core.structs.event import Event
from core.structs.incident import Incident
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class ActorMachine:
    """
    State Machine for a spill in a given camera. Note that all spills are treated as one spill.
    """

    MAX_RANDOM_SPILL_RAISE_THRESHOLD_S = 3600  # 60 mins

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ) -> None:
        self.camera_uuid = camera_uuid
        self.actor_info = actor_info

        if params is not None:
            # trunk-ignore(pylint/C0103)
            self.MAX_RANDOM_SPILL_RAISE_THRESHOLD_S = params.get(
                "max_random_spill_raise_threshold_s",
                self.MAX_RANDOM_SPILL_RAISE_THRESHOLD_S,
            )

            # trunk-ignore(pylint/C0103)
            self.MAX_RANDOM_SPILL_RAISE_THRESHOLD_MS = (
                self.MAX_RANDOM_SPILL_RAISE_THRESHOLD_S * 1000
            )

        self.last_spill_raised_timestamp_ms = 0
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None

        self._initialize_state_machine()

    def _initialize_state_machine(self) -> None:
        states = ["start", "incident"]

        transitions = [
            {
                "trigger": "message_received",
                "source": "start",
                "dest": "incident",
                "conditions": "_max_incident_threshold_passed",
            },
            {
                "trigger": "message_received",
                "source": "incident",
                "dest": None,
            },
            {
                "trigger": "try_reset_state",
                "source": "incident",
                "dest": "start",
            },
            {
                "trigger": "try_reset_state",
                "source": "start",
                "dest": None,
            },
        ]

        Machine(
            model=self,
            states=states,
            transitions=transitions,
            initial="start",
            ignore_invalid_triggers=False,
        )

    def transition_state_event(
        self, state_event_message: Union[State, Event]
    ) -> Optional[Incident]:
        """Feed the state-event message to the state machine and transition it.

        Args:
            state_event_message (Union[State, Event]): Info about either
            the State or the Event.

        Returns:
            Optional[Incident]: returns the generated_incident
        """
        incident_list = []
        self.state_event_start_timestamp_ms = state_event_message.timestamp_ms
        self.state_event_end_timestamp_ms = (
            state_event_message.end_timestamp_ms
        )

        self.message_received(
            incident_list=incident_list,
        )

        self.try_reset_state()

        return incident_list[0] if len(incident_list) > 0 else None

    # Conditionals
    def _max_incident_threshold_passed(self, **kwargs: dict) -> bool:
        """Checks whether the time threshold to raise an incident has passed.

        Returns:
            bool: whether the time threshold to raise an incident has passed.
        """
        return (
            self.state_event_end_timestamp_ms
            - self.last_spill_raised_timestamp_ms
            >= self.MAX_RANDOM_SPILL_RAISE_THRESHOLD_MS
        )

    def on_enter_incident(self, incident_list: List) -> None:
        """This function is called whenever the transition machine enters the
        incident state. We raise an incident everytime the FSM enters this state
        and sets the time of incident raise.

        Args:
            incident_list: list to populate with the generated incident
        """
        incident_list.append(
            Incident(
                camera_uuid=self.camera_uuid,
                start_frame_relative_ms=self.state_event_start_timestamp_ms,
                end_frame_relative_ms=self.state_event_start_timestamp_ms,
                actor_ids=[self.actor_info.actor_id],
                track_uuid=self.actor_info.track_uuid,
            )
        )
        self.last_spill_raised_timestamp_ms = self.state_event_end_timestamp_ms


class SpillMachine(BaseStateMachine):
    """Interface class for per actor state machines"""

    NAME = "random_spill"
    INCIDENT_TYPE_NAME = "Spill"
    INCIDENT_TYPE_ID = "SPILL"
    PRE_START_BUFFER_MS = 1000
    POST_END_BUFFER_MS = 1000
    INCIDENT_VERSION = "experimental-0.1"
    INCIDENT_PRIORITY = "high"

    MAX_CACHE_SIZE = 1
    MAX_CACHE_TIME_S = 24 * 60 * 60  # 24 hrs

    # No Cooldowns for random spill generator as the machine
    # maintains its own internal clock and only raises at fixed intervals.
    PER_ACTOR_COOLDOWN_THRESHOLD_S = 0

    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        """
        This is a hack. We basically want to update spill machine state for every event
        and state messages generated. This acts as sort of a heartbeat for this machine.
        A caveat is that this machine will not generate any incidents if no states and
        events are being generated at all. That is a reasonable assumption for now.

        Args:
            state_event_message: message containing a state or event

        Returns:
            A fake actor ID that enables us to generate an incident for every state_event_message
        """
        return RelevantActorInfo(
            actor_id="spill_actor",
            track_uuid="spill_actor",
        )

    def transition_actor_machine(
        self,
        actor_info: RelevantActorInfo,
        state_event_message: Union[State, Event],
    ) -> Optional[Incident]:
        """
        An incident is generated when the maximum time threshold for raising incident is met.
        This machine uses state and event messages as heartbeat.

        Args:
            actor_info: identifiers of a fake actor spill_actor
            state_event_message: message containing a state or event

        Returns:
            Optional incident when the maximum time threshold for raising incident is met.
        """

        if actor_info.track_uuid not in self.actor_machines:
            # create the machine first
            self.actor_machines[actor_info.track_uuid] = ActorMachine(
                self.camera_uuid, actor_info, params=self.params
            )

        actor_machine = self.actor_machines[actor_info.track_uuid]
        return actor_machine.transition_state_event(state_event_message)
