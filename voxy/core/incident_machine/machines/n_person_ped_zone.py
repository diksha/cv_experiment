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
    max_ped_zone_duration_s = 10
    max_persons_in_ped_zone = 1

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ) -> None:
        self.actor_info = actor_info
        self.camera_uuid = camera_uuid

        if params is not None:
            self.max_ped_zone_duration_s = params.get(
                "max_ped_zone_duration_s", self.max_ped_zone_duration_s
            )
            self.max_persons_in_ped_zone = params.get(
                "max_persons_in_ped_zone", self.max_persons_in_ped_zone
            )

        self.max_ped_zone_duration_ms = self.max_ped_zone_duration_s * 1000
        self.first_time_threshold_exceeded = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None
        self._is_in_cooldown = False
        self.sequence_id = 0

        self._initialize_state_machine()

    def _initialize_state_machine(self) -> None:
        states = ["start", "more_persons_than_threshold", "incident"]

        transitions = [
            {
                "trigger": "update_zone_state",
                "source": ["start"],
                "dest": "more_persons_than_threshold",
                "conditions": [
                    "_more_persons_than_threshold_in_ped_zone",
                ],
            },
            {
                "trigger": "update_zone_state",
                "source": ["more_persons_than_threshold", "incident"],
                "conditions": [
                    "_more_persons_than_threshold_in_ped_zone",
                ],
                "dest": None,
            },
            {
                "trigger": "update_zone_state",
                "source": ["start", "more_persons_than_threshold", "incident"],
                "unless": [
                    "_more_persons_than_threshold_in_ped_zone",
                ],
                "dest": "start",
            },
            {
                "trigger": "check_incident",
                "source": "more_persons_than_threshold",
                "conditions": [
                    "_more_persons_than_threshold_for_greater_than_time_threshold",
                ],
                "dest": "incident",
                "after": "_get_incident",
            },
            {
                "trigger": "check_incident",
                "source": ["start", "more_persons_than_threshold", "incident"],
                "dest": None,
            },
            {
                "trigger": "try_reset_state",
                "source": "incident",
                "dest": "more_persons_than_threshold",
            },
            {
                "trigger": "try_reset_state",
                "source": [
                    "start",
                    "more_persons_than_threshold",
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
        """Transitions the incident machine states.

        Args:
            state_event_message (Union[State, Event]): Next state or event message to transition on.

        Returns:
            Optional[Incident]: Returns a list of all generated incidents.
        """
        incident_list = []
        self.state_event_start_timestamp_ms = state_event_message.timestamp_ms
        self.state_event_end_timestamp_ms = (
            state_event_message.end_timestamp_ms
        )

        self.update_zone_state(
            num_persons_in_no_ped_zone=state_event_message.num_persons_in_no_ped_zone,
        )

        self.check_incident(incident_list=incident_list)

        self.try_reset_state()

        return incident_list[0] if len(incident_list) > 0 else None

    def _more_persons_than_threshold_in_ped_zone(
        self, num_persons_in_no_ped_zone, **kwargs
    ) -> bool:
        """Returns if the number of persons in a No Ped Zone is greater than the
        threshold.

        Args:
            num_persons_in_no_ped_zone (int): int of number of persons in ped zone

        Returns:
            bool: whether the number of persons exceeds the threshold.
        """
        return num_persons_in_no_ped_zone > self.max_persons_in_ped_zone

    def _more_persons_than_threshold_for_greater_than_time_threshold(
        self, **kwargs
    ) -> bool:
        """Returns if the machiine has been in the more_persons_than_threshold
        state for greater than time threshold.

        Returns:
            bool: true if time threshold is reached.
        """
        return (
            self.state_event_end_timestamp_ms
            - self.first_time_threshold_exceeded
            > self.max_ped_zone_duration_ms
        )

    # Callbacks
    def _get_incident(self, incident_list, **kwargs) -> None:
        """A list of incidetnts generated.

        Args:
            incident_list (list): list of incidents
        """
        incident_list.append(
            Incident(
                camera_uuid=self.camera_uuid,
                start_frame_relative_ms=self.first_time_threshold_exceeded,
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

    def on_enter_more_persons_than_threshold(self, **kwargs) -> None:
        """Sets the timestamp when entering new state.
        Args:
            **kwargs: kwargs
        """
        self.first_time_threshold_exceeded = (
            self.state_event_start_timestamp_ms
        )

    def on_enter_start(self, **kwargs) -> None:
        """Resets the timestamps on starting the incident machine.
        Args:
            **kwargs: kwargs
        """
        # reset everything
        self.first_time_threshold_exceeded = None
        self.state_event_start_timestamp_ms = None
        self.state_event_end_timestamp_ms = None
        self._is_in_cooldown = False
        self.sequence_id += 1


class NPersonPedZoneMachine(BaseStateMachine):
    """TODO(harishma): Add a comment documenting the caveats with this machine and how it differs
    from all other incident machines which deal with dynamic actors.
    """

    NAME = "n_person_ped_zone"
    INCIDENT_TYPE_NAME = "More than one pedestrian in No-Ped Zone"
    INCIDENT_TYPE_ID = "N_PERSON_PED_ZONE"
    PRE_START_BUFFER_MS = 1000
    POST_END_BUFFER_MS = 2000
    INCIDENT_VERSION = "1.0"
    INCIDENT_PRIORITY = "medium"

    # This machine by default should not suppress incidents by actor id
    # as the actor is no ped zone. Because it's a static actor the id never changes.
    PER_ACTOR_COOLDOWN_THRESHOLD_S = 0
    PER_ACTOR_COOLDOWN_THRESHOLD_MS = PER_ACTOR_COOLDOWN_THRESHOLD_S * 1000

    # Cache Configs
    MAX_CACHE_SIZE = 50  # max 50 actors in a scene
    MAX_CACHE_TIME_S = 10 * 60  # 10 minute

    def get_relevant_actor(
        self, state_event_message: Union[State, Event]
    ) -> Optional[RelevantActorInfo]:
        """Returns the identifiers of the relevant actor

        Args:
            state_event_message (Union[State, Event]): current state or event message

        Returns:
            Optional[str]: Relevant actor info or None.
        """
        if (
            isinstance(state_event_message, State)
            and state_event_message.actor_category == ActorCategory.NO_PED_ZONE
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
        """Transitions each actor machine

        Args:
            actor_info (RelevantActorInfo): relevant actor identifiers
            state_event_message (Union[State, Event]): current state or event message

        Returns:
            Optional[Incident]: List of incidents.
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
