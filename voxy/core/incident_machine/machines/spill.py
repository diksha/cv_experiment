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
from core.structs.event import Event, EventType
from core.structs.incident import Incident
from core.structs.state import State


class ActorStateMachine:
    """
    State Machine for a spill in a given camera. Note that all spills are treated as one spill.
    """

    MAX_INCIDENT_RESET_TIME_S = 1800  # 30 mins

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ) -> None:
        self.camera_uuid = camera_uuid
        self.actor_info = actor_info
        self.spill_actor = None

        if params is not None:
            # trunk-ignore(pylint/C0103)
            self.MAX_INCIDENT_RESET_TIME_S = params.get(
                "max_incident_reset_time_s",
                self.MAX_INCIDENT_RESET_TIME_S,
            )

        # trunk-ignore(pylint/C0103)
        self.MAX_INCIDENT_RESET_TIME_MS = self.MAX_INCIDENT_RESET_TIME_S * 1000

        self.last_spill_detected_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None

        self._initialize_state_machine()

    def _initialize_state_machine(self) -> None:
        """
        Create and initiation a state machine to track states of a spill in a camera.
        The state machine recieves transition requests if and only if a spill detected
        event occurs. Spill detected events are continous in nature which means multiple
        spill detected incidents will be raised for the same spill untill it exists. We
        reset the machine when a spill detected incident hasn't been seen for a period of
        time given by MAX_INCIDENT_RESET_TIME_MS.
        """
        states = ["start", "incident"]

        transitions = [
            {
                "trigger": "spill_detected",
                "source": "start",
                "dest": "incident",
                "after": "_register_spill_detected",
            },
            {
                "trigger": "spill_detected",
                "source": ["incident"],
                "dest": "incident",
                "conditions": "_incident_reset_threshold_passed",
                "after": "_register_spill_detected",
            },
            {
                "trigger": "spill_detected",
                "source": ["incident"],
                "dest": None,
                "after": "_register_spill_detected",
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
        """
        Transition a given state machine according to a state or event message.

        Args:
            state_event_message: State or Event message

        Returns:
            An incident if generated.
        """
        incident_list = []
        self.spill_actor = RelevantActorInfo(
            actor_id=state_event_message.object_id,
            track_uuid=state_event_message.object_uuid,
        )
        self.state_event_start_timestamp_ms = state_event_message.timestamp_ms
        self.state_event_end_timestamp_ms = (
            state_event_message.end_timestamp_ms
        )

        # trunk-ignore(pylint/E1101)
        self.spill_detected(
            incident_list=incident_list,
        )

        return incident_list[0] if len(incident_list) > 0 else None

    # Conditionals
    def _incident_reset_threshold_passed(self, **kwargs: dict) -> bool:
        """
        Calculate whether a certain amount of time has passed since spill was last seen.
        If the threshold has passed then we can resume raising an incident when a spill.
        is detected.

        Returns:
            Whether a certain amount of time has passed since we entered incident state.
        """
        return (
            self.state_event_start_timestamp_ms
            - self.last_spill_detected_timestamp_ms
            > self.MAX_INCIDENT_RESET_TIME_MS
        )

    # Callbacks
    def _register_spill_detected(self, **kwargs: dict) -> None:
        self.last_spill_detected_timestamp_ms = (
            self.state_event_end_timestamp_ms
        )

    def on_enter_incident(self, incident_list: List) -> None:
        """
        Callback that is invoked when we enter/re-enter the incident state.

        Args:
            incident_list: Any newly generated incident is added to this list.
        """
        incident_list.append(
            Incident(
                camera_uuid=self.camera_uuid,
                start_frame_relative_ms=self.state_event_start_timestamp_ms,
                end_frame_relative_ms=self.state_event_end_timestamp_ms,
                actor_ids=[]
                if self.spill_actor is None
                else [self.spill_actor.actor_id],
                track_uuid=None
                if self.spill_actor is None
                else self.spill_actor.track_uuid,
            )
        )


class SpillMachine(BaseStateMachine):
    """Interface class for per actor state machines"""

    NAME = "spill"
    INCIDENT_TYPE_NAME = "Spill"
    INCIDENT_TYPE_ID = "SPILL"
    PRE_START_BUFFER_MS = 1000
    POST_END_BUFFER_MS = 1000
    INCIDENT_VERSION = "1.0"
    INCIDENT_PRIORITY = "high"

    MAX_CACHE_SIZE = 1
    MAX_CACHE_TIME_S = 60 * 1  # 1 min

    # The length of camera cooldwon is long and will suppress spills from different
    # actors generated within this cooldown period. This is the desired behaviour as we do not want
    # the customer to be notified more than once of a spill in a particular room even if
    # it may be in a new location (different actor)

    PER_CAMERA_COOLDOWN_THRESHOLD_S = (
        60 * 60
    )  # Raise incident only 60 minutes apart

    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        """
        Get id of the actor that this state machine cares
        about from a given state or event message

        Args:
            state_event_message: message containing information
                                 about the current state or event

        Returns:
            The id of an actor if a relevant actor exists
        """
        if (
            isinstance(state_event_message, Event)
            and state_event_message.event_type == EventType.SPILL_DETECTED
        ):
            return RelevantActorInfo(
                actor_id=state_event_message.subject_id,
                track_uuid=state_event_message.subject_uuid,
            )

        return None

    def transition_actor_machine(
        self,
        actor_info: RelevantActorInfo,
        state_event_message: Union[State, Event],
    ) -> Optional[Incident]:
        """
        Transition the state machine of an actor identified by track_uuid
        with the message in state_event_message

        Args:
            actor_info: info of the actor whose state machine needs to be transitioned.
            state_event_message: message containing information
                                 about the current state or event

        Returns:
            An incident if any were generated by the transition of the state machine
        """
        if actor_info.track_uuid not in self.actor_machines:
            # create the machine first
            self.actor_machines[actor_info.track_uuid] = ActorStateMachine(
                self.camera_uuid,
                actor_info,
                params=self.params,
            )

        actor_machine = self.actor_machines[actor_info.track_uuid]
        return actor_machine.transition_state_event(state_event_message)
