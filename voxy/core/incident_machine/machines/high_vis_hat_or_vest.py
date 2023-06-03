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
from core.structs.event import Event
from core.structs.incident import Incident
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class ActorMachine:
    # Atleast 5 sec with no hard hat and vest to be considered a violation
    max_no_hat_or_vest_s = 5

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ):
        self.camera_uuid = camera_uuid
        self.actor_info = actor_info

        if params is not None:
            self.max_no_hat_or_vest_s = params.get(
                "max_no_hat_or_vest_s", self.max_no_hat_or_vest_s
            )

        self.max_no_hat_or_vest_ms = self.max_no_hat_or_vest_s * 1000

        self.first_seen_not_wearing_hat_or_vest_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None

        self._initialize_state_machine()

    def _initialize_state_machine(self):
        states = ["start", "not_wearing_hat_or_vest_on_floor", "incident"]

        transitions = [
            {
                "trigger": "update_person_state",
                "source": "start",
                "dest": "not_wearing_hat_or_vest_on_floor",
                "conditions": [
                    "_is_not_wearing_hat_or_vest",
                    "_is_not_associated",
                ],
            },
            # If we have already raised an incident then don't ever raise again for the same
            # actor unless they fail any of the condition which will put them back in start.
            {
                "trigger": "update_person_state",
                "source": ["not_wearing_hat_or_vest_on_floor", "incident"],
                "conditions": [
                    "_is_not_wearing_hat_or_vest",
                    "_is_not_associated",
                ],
                "dest": None,
            },
            {
                "trigger": "update_person_state",
                "source": "*",
                "dest": "start",
            },
            {
                "trigger": "check_incident",
                "source": "not_wearing_hat_or_vest_on_floor",
                "conditions": [
                    "_is_not_wearing_for_greater_than_time_threshold",
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
                "dest": "start",
            },
            {
                "trigger": "try_reset_state",
                "source": [
                    "start",
                    "not_wearing_hat_or_vest_on_floor",
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
        conditions (eg. person_is_wearing_hard_hat (or vest), person_is_associated, etc) and
        their respective timestamps (Eg. state_event_start, state_event_end, etc).
        Also, adds _get_incident to the incident_list if source is
        not_wearing_hat_or_vest_on_floor and the time threshold condition satisifies.

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
            person_is_wearing_hard_hat=state_event_message.person_is_wearing_hard_hat,
            person_is_wearing_safety_vest=state_event_message.person_is_wearing_safety_vest,
            person_is_associated=state_event_message.person_is_associated,
        )

        self.check_incident(incident_list=incident_list)

        self.try_reset_state()

        return incident_list[0] if len(incident_list) > 0 else None

    def _is_not_wearing_for_greater_than_time_threshold(
        self, **kwargs
    ) -> bool:
        """Checks if the time threshold is reached for not_wearing_hat_or_vest_on_floor

        Returns:
            bool: True if time > threshold else False
        """
        return (
            self.state_event_end_timestamp_ms
            - self.first_seen_not_wearing_hat_or_vest_timestamp_ms
            > self.max_no_hat_or_vest_ms
        )

    def _is_not_wearing_hat_or_vest(
        self,
        person_is_wearing_hard_hat,
        person_is_wearing_safety_vest,
        **kwargs,
    ) -> bool:
        """Check if the person is wearing safety vest and hard hat

        Args:
            person_is_wearing_hard_hat (bool): Checks if person is wearing hard hat (bool)
            person_is_wearing_safety_vest (bool): Checks if person is wearing safety vest (bool)

        Returns:
            bool: True if not wearing hard hat and safety vest else False
        """
        # Check only False and ignore None
        return (
            person_is_wearing_hard_hat is False
            and person_is_wearing_safety_vest is False
        )

    def _is_not_associated(self, person_is_associated, **kwargs):
        """Check if the person is *NOT* associated to a PIT

        Args:
            person_is_associated (bool): True if associated to a PIT

        Returns:
            bool: return True if not associated else False
        """
        # Check None and False both.
        return not person_is_associated

    # Callbacks
    def _get_incident(self, incident_list, **kwargs) -> None:
        incident_list.append(
            Incident(
                camera_uuid=self.camera_uuid,
                start_frame_relative_ms=self.first_seen_not_wearing_hat_or_vest_timestamp_ms,
                end_frame_relative_ms=self.state_event_end_timestamp_ms,
                actor_ids=[self.actor_info.actor_id],
                track_uuid=self.actor_info.track_uuid,
            )
        )

    def on_enter_not_wearing_hat_or_vest_on_floor(self, **_) -> None:
        """This function is called whenever the transition machine enters
        the on_enter_not_wearing_hat_or_vest_on_floor state and we note the
        respective state_event_timestamp

        Args:
            **_: kwargs
        """
        self.first_seen_not_wearing_hat_or_vest_timestamp_ms = (
            self.state_event_start_timestamp_ms
        )

    def on_enter_start(self, **_) -> None:
        """This function is called whenever the transition machine enters
        the start state and we restart all the timestamps and our machine

        Args:
            **_: kwargs
        """
        # reset everything
        self.first_seen_not_wearing_hat_or_vest_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None


class HighVisHatOrVestViolationMachine(BaseStateMachine):
    # State Machine Diagram:
    # https://drive.google.com/file/d/19dXhssHr7EqXNiW5A-Cxqh7lxSYzBoSS/view?usp=share_link
    """Modify the BaseStateMachine by providing specific info on incidents, cache,
    and defining abstract methods like get_relevant_actor and transition_actor_machine.

    Args:
        BaseStateMachine (_type_): Calls this class from base.py which has
        all the info on incidents, cooldown, cache and it processes all the
        class parameters.
    """

    NAME = "high_vis_hat_or_vest"
    INCIDENT_TYPE_ID = "HIGH_VIS_HAT_OR_VEST"
    INCIDENT_TYPE_NAME = "High Vis Hat or Vest"
    PRE_START_BUFFER_MS = 9000
    POST_END_BUFFER_MS = 6000
    INCIDENT_VERSION = "1.0"
    INCIDENT_PRIORITY = "low"

    # Cache Configs
    MAX_CACHE_SIZE = 50  # max 50 actors in a scene
    MAX_CACHE_TIME_S = 10 * 60  # 10 minutes

    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        """Function to return the actor info associated with the state_event_message
        that the BaseStateMachine process to get the generated_incident and
        if it is greater than cooldown time, adds incident to the incident_list.

        Args:
            state_event_message (Union[State, Event]): from core/structs/protobufs
            State - It is the current scenario associated to an actor
            Eg - door_is_open, person_wearing_hat, etc.
            Event - It is an activity that will change the state of an actor
            Eg - Door_Opening, PIT_Exiting_Door, etc.

        Returns:
            Optional[str]: actor_info of the state_event_message
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
            actor_info (RelevantActorInfo): Actor unique identification
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
