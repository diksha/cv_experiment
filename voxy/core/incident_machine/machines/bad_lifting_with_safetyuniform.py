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

from typing import Optional, Union

from transitions import Machine

from core.incident_machine.machines.base import (
    BaseStateMachine,
    RelevantActorInfo,
)
from core.structs.actor import ActorCategory
from core.structs.ergonomics import PostureType
from core.structs.event import Event
from core.structs.incident import Incident
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class ActorMachine:
    """
    This ActorMachine class defines all the parameters and
    functions required for a specific transition machine
    """

    # Configs: Initialized values from bad_lifting.py and safetyvest.py
    # Note: Bad_lift is used interchangably with bad_posture
    max_allowed_bad_lift_with_vest_duration_s = 0.5  # Using min of the two

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Optional[dict] = None,
    ) -> None:
        self.camera_uuid = camera_uuid
        self.actor_info = actor_info

        if params is not None:
            self.max_allowed_bad_lift_with_vest_duration_s = params.get(
                "max_allowed_bad_lift_with_vest_duration_s",
                self.max_allowed_bad_lift_with_vest_duration_s,
            )

        self.max_allowed_bad_lift_with_vest_duration_ms = (
            self.max_allowed_bad_lift_with_vest_duration_s * 1000
        )

        self.first_seen_not_bad_lifting_with_vest_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None

        self._intialize_state_machine()

    def _intialize_state_machine(self) -> None:
        states = [
            "start",
            "lifting_badly_with_wearing_uniform_on_floor",  # on floor means not associated with PIT
            "incident",
        ]

        transitions = [
            {
                "trigger": "update_person_state",
                "source": "start",
                "dest": "lifting_badly_with_wearing_uniform_on_floor",
                "conditions": [
                    "_is_lifting_badly",
                    "_is_wearing_safety_vest",
                    "_is_not_associated",
                ],
            },
            {
                "trigger": "update_person_state",
                "source": [
                    "lifting_badly_with_wearing_uniform_on_floor",
                    "incident",
                ],
                "dest": None,
                "conditions": [
                    "_is_lifting_badly",
                    "_is_wearing_safety_vest",
                    "_is_not_associated",
                ],
            },
            # Restart the FSM if any of the above 3 conditions fail
            {
                "trigger": "update_person_state",
                "source": "*",
                "dest": "start",
            },
            {
                "trigger": "check_incident",
                "source": "lifting_badly_with_wearing_uniform_on_floor",
                "dest": "incident",
                "conditions": [
                    "_is_lifting_badly_with_vest_for_greater_than_time_threshold",
                ],
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
                "dest": "start",
            },
            {
                "trigger": "try_reset_state",
                "source": [
                    "start",
                    "lifting_badly_with_wearing_uniform_on_floor",
                ],
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
        """Contains the attributes required to check for the
        conditions (eg. person_lift_type, person_is_associated, etc) and
        their respective timestamps (Eg. state_event_start, state_event_end, etc).
        Also, adds _get_incident to the incident_list if source is
        lifting_badly_with_wearing_vest_on_floor and time threshold satisifies.

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

        self.update_person_state(
            person_lift_type=state_event_message.person_lift_type,
            person_is_wearing_safety_vest=state_event_message.person_is_wearing_safety_vest,
            person_is_associated=state_event_message.person_is_associated,
        )

        self.check_incident(incident_list=incident_list)

        self.try_reset_state()

        return incident_list[0] if len(incident_list) > 0 else None

    def _is_lifting_badly_with_vest_for_greater_than_time_threshold(
        self, **_
    ) -> bool:
        """Checks if the threshold is reached for bad lifting with vest

        Returns:
            bool: True if time > threshold else False
        """
        return (
            self.state_event_end_timestamp_ms
            - self.first_seen_not_bad_lifting_with_vest_timestamp_ms
            > self.max_allowed_bad_lift_with_vest_duration_ms
        )

    def _is_lifting_badly(self, person_lift_type, **_) -> bool:
        """Checks if the actor (person) is lifting badly (bad posture)

        Args:
            person_lift_type (_type_): used to check if person_lift_type (int)
            is bad_posture (which is 0 in structs/ergonomics)

        Returns:
            bool: True if lifting_badly (person_lift_type == 0) else False
        """
        # Check only False and ignore None
        return person_lift_type == PostureType.BAD

    def _is_wearing_safety_vest(
        self, person_is_wearing_safety_vest, **_
    ) -> bool:
        """Check if the person is wearing safety vest

        Args:
            person_is_wearing_safety_vest (_type_): Tells whether person is wearing
            safety vest or not (bool)

        Returns:
            bool: True if wearing safety vest else False
        """
        # Check only True and ignore None
        return person_is_wearing_safety_vest is True

    def _is_not_associated(self, person_is_associated, **_) -> bool:
        """Check if the person is *NOT* associated to a PIT

        Args:
            person_is_associated (bool): True if associated to a PIT

        Returns:
            bool: return True if not associated else False
        """
        # Check None and False both.
        return not person_is_associated

    # Callbacks
    def _get_incident(self, incident_list, **_) -> None:
        incident_list.append(
            Incident(
                camera_uuid=self.camera_uuid,
                start_frame_relative_ms=self.first_seen_not_bad_lifting_with_vest_timestamp_ms,
                end_frame_relative_ms=self.state_event_end_timestamp_ms,
                actor_ids=[self.actor_info.actor_id],
                track_uuid=self.actor_info.track_uuid,
            )
        )

    def on_enter_lifting_badly_with_wearing_uniform_on_floor(
        self, **_
    ) -> None:
        """This function is called whenever the transition machine enters
        the lifting_badly_with_wearing_uniform_on_floor state and we note the
        respective state_event_timestamp

        Args:
            **_: kwargs
        """
        self.first_seen_not_bad_lifting_with_vest_timestamp_ms = (
            self.state_event_start_timestamp_ms
        )

    def on_enter_start(self, **_) -> None:
        """This function is called whenever the transition machine enters
        the start state and we restart all the timestamps and our machine

        Args:
            **_: kwargs
        """
        self.first_seen_not_bad_lifting_with_vest_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None


class BadLiftingWithSafetyUniformViolationMachine(BaseStateMachine):
    # State Machine Diagram:
    # https://drive.google.com/file/d/1YGfZYDRgLp7NQDB2X0jSKjFDQXx9d63b/view?usp=sharing
    """Modify the BaseStateMachine by providing specific info on incidents, cache,
    and defining abstract methods like get_relevant_actor and transition_actor_machine.

    Args:
        BaseStateMachine (_type_): Calls this class from base.py which has
        all the info on incidents, cooldown, cache and it processes all the
        class parameters.
    """

    NAME = "bad_posture_with_uniform"
    INCIDENT_TYPE_ID = "BAD_POSTURE_WITH_SAFETY_UNIFORM"
    INCIDENT_TYPE_NAME = "Bad Posture with Safety Uniform"
    PRE_START_BUFFER_MS = 10000
    POST_END_BUFFER_MS = 10000
    INCIDENT_VERSION = "0.1"
    INCIDENT_PRIORITY = "low"

    # Cache Configs
    MAX_CACHE_SIZE = 50  # max 50 actors in a scene
    MAX_CACHE_TIME_S = 10 * 60  # 10 minute

    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        """Function to return the actor_info associated with the state_event_message
        that the BaseStateMachine process to get the generated_incident and
        if it is greater than cooldown time, adds incident to the incident_list.

        Args:
            state_event_message (Union[State, Event]): from core/structs/protobufs
            State - It is the current scenario associated to an actor
            Eg - door_is_open, person_wearing_hat, etc.
            Event - It is an activity that will change the state of an actor
            Eg - Door_Opening, PIT_Exiting_Door, etc.

        Returns:
            Optional[RelevantActorInfo]: actor_info of the state_event_message
        """
        if (
            isinstance(state_event_message, State)
            and state_event_message.actor_category == ActorCategory.PERSON
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
        """Triggers the transition machine and returns the generated incident

        Args:
            actor_info (str): Actor info containing unique identification information
            state_event_message (Union[State, Event]): Info about either the State
            or the Event.

        Returns:
            Optional[Incident]: generated_incident to the corresponding actor
        """

        if actor_info.track_uuid not in self.actor_machines:
            # create the machine first
            self.actor_machines[actor_info.track_uuid] = ActorMachine(
                self.camera_uuid,
                actor_info,
                params=self.params,
            )

        actor_machine = self.actor_machines[actor_info.track_uuid]
        return actor_machine.transition_state_event(state_event_message)
