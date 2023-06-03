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
from core.structs.event import Event
from core.structs.incident import Incident
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class ActorMachine:
    """
    This ActorMachine class defines all the parameters like max_no_motion_detection_s
    and the transition machine states and conditions
    """

    max_no_motion_detection_s = 60
    max_no_motion_detection_ms = max_no_motion_detection_s * 1000

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ) -> None:
        self.camera_uuid = camera_uuid
        self.actor_info = actor_info

        if params is not None:
            self.max_no_motion_detection_s = params.get(
                "max_no_motion_detection_s", self.max_no_motion_detection_s
            )
            self.max_no_motion_detection_ms = (
                self.max_no_motion_detection_s * 1000
            )

        self.last_known_unalerted_downtime_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None
        self._is_in_cooldown = False
        self.sequence_id = 0

        self._initialize_state_machine()

    def _initialize_state_machine(self) -> None:
        states = ["motion", "downtime", "incident"]

        transitions = [
            {
                "trigger": "update_motion_state",
                "source": "motion",
                "dest": "downtime",
                "conditions": "_is_frozen",
                "after": "_record_first_known_downtime_timestamp",
            },
            {
                "trigger": "update_motion_state",
                "source": ["downtime", "incident"],
                "dest": None,
                "conditions": "_is_frozen",
            },
            {
                "trigger": "update_motion_state",
                "source": "*",
                "dest": "motion",
            },
            {
                "trigger": "check_incident",
                "source": "downtime",
                "conditions": [
                    "_is_frozen_for_greater_than_time_threshold",
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
                "dest": "downtime",
                "after": "_record_downtime_reset_timestamp",
            },
            {
                "trigger": "try_reset_state",
                "source": [
                    "motion",
                    "downtime",
                ],
                "dest": None,
            },
        ]
        Machine(
            model=self,
            states=states,
            transitions=transitions,
            initial="motion",
            ignore_invalid_triggers=False,
        )

    def transition_state_event(
        self, state_event_message: Union[State, Event]
    ) -> Optional[Incident]:
        """Contains the function update motion state and check incident triggers
        and their respective timestamps (Eg. state_event_start, state_event_end, etc).
        Also, adds _get_incident to the incident_list if source is
        downtime and threshold condition satisifies.

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

        self.update_motion_state(
            is_in_motion=state_event_message.motion_zone_is_in_motion
        )

        self.check_incident(incident_list=incident_list)

        self.try_reset_state()

        return incident_list[0] if len(incident_list) > 0 else None

    # Conditionals
    def _is_frozen_for_greater_than_time_threshold(self, **kwargs) -> bool:
        """Checks if the threshold is reached for downtime (frozen state)

        Returns:
            bool: True if time > threshold else False
        """
        return (
            self.state_event_end_timestamp_ms
            - self.last_known_unalerted_downtime_timestamp_ms
            > self.max_no_motion_detection_ms
        )

    def _is_frozen(self, is_in_motion, **kwargs) -> bool:
        """Checks if the actor (MotionDetectionZoneState) is frozen

        Args:
            motion_state (_type_): used to check if motion_state (int)
            is frozen (which is 1 in actor.py)

        Returns:
            bool: True if frozen (frozen == 1) else False
        """
        return not is_in_motion

    def _record_downtime_reset_timestamp(self, **kwargs) -> None:
        """When entering downtime from incident state, set the last known
        unalerted time to the end of the current state message.
        """
        self.last_known_unalerted_downtime_timestamp_ms = (
            self.state_event_end_timestamp_ms
        )

    def _record_first_known_downtime_timestamp(self, **kwargs) -> None:
        """When entering downtime from the motion state, set the last known
        unalerted time to the start of the current state message as no incident has
        been generated yet.
        """
        self.last_known_unalerted_downtime_timestamp_ms = (
            self.state_event_start_timestamp_ms
        )

    # Callbacks
    def _get_incident(self, incident_list, **kwargs) -> None:
        incident_list.append(
            Incident(
                camera_uuid=self.camera_uuid,
                start_frame_relative_ms=self.last_known_unalerted_downtime_timestamp_ms,
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

    def on_enter_motion(self, **kwargs) -> None:
        """This function is called whenever the transition machine enters the
        motion (start) state and we restart all the timestamps and our machine

        Args:
            **kwargs: kwargs
        """
        self.last_known_unalerted_downtime_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None
        self._is_in_cooldown = False
        self.sequence_id += 1


class MotionDetectionDowntimeMachine(BaseStateMachine):
    # State Machine Diagram:
    # https://drive.google.com/file/d/1bh8GRmB082-TlOszrFa6qGaYkuAVe6xb/view?usp=sharing
    """TODO(harishma): Add a comment documenting the caveats with this machine and how it differs
    from all other incident machines which deal with dynamic actors.

    Modify the BaseStateMachine by providing specific info on incidents, cache,
    and defining abstract methods like get_relevant_actor (MOTION_DETECTION_ZONE)
    and transition_actor_machine.

    Args:
        BaseStateMachine (_type_): Calls this class from base.py which has
        all the info on incidents, cooldown, cache and it processes all the
        class parameters.
    """

    NAME = "production_line_down"
    INCIDENT_TYPE_NAME = "Production Line Down"
    INCIDENT_TYPE_ID = "PRODUCTION_LINE_DOWN"
    PRE_START_BUFFER_MS = 60000
    POST_END_BUFFER_MS = 5000
    INCIDENT_VERSION = "0.1"
    INCIDENT_PRIORITY = "medium"

    # This machine by default should not suppress incidents by actor id
    # as the actor is motion detection zone. Because it's a static actor the id never changes.
    PER_ACTOR_COOLDOWN_THRESHOLD_S = 0
    PER_ACTOR_COOLDOWN_THRESHOLD_MS = PER_ACTOR_COOLDOWN_THRESHOLD_S * 1000

    # Cache Configs
    MAX_CACHE_SIZE = 5  # max 5 actors in a scene
    MAX_CACHE_TIME_S = 12 * 60 * 60  # 12 hours

    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        """Function to return the actor_info associated with the state_event_message
        that the BaseStateMachine process to get the generated_incident and
        if it is greater than cooldown time, adds incident to the incident_list.

        Args:
            state_event_message (Union[State, Event]): from core/structs/protobufs
            State - It is the current scenario associated to an actor
            Eg - motion_detection_zone_state.
            Event - It is an activity that will change the state of the belt

        Returns:
            Optional[RelevantActorInfo]: actor_info of the state_event_message
        """
        if (
            isinstance(state_event_message, State)
            and state_event_message.actor_category
            == ActorCategory.MOTION_DETECTION_ZONE
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
