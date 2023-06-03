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

from cachetools import TTLCache
from transitions import Machine

from core.incident_machine.machines.base import (
    BaseStateMachine,
    RelevantActorInfo,
)
from core.structs.actor import ActorCategory, DoorState
from core.structs.event import Event, EventType
from core.structs.incident import Incident
from core.structs.state import State


# trunk-ignore-all(pylint/E1101)
class ActorMachine:
    MAX_PIT_CROSSING_TIME_DIFFERENCE_S = 30

    def __init__(
        self,
        camera_uuid: str,
        actor_info: RelevantActorInfo,
        params: Union[dict, None] = None,
    ) -> None:
        self.camera_uuid = camera_uuid
        self.actor_info = actor_info

        if params is not None:
            self.MAX_PIT_CROSSING_TIME_DIFFERENCE_S = params.get(
                "max_pit_crossing_time_differences_s",
                self.MAX_PIT_CROSSING_TIME_DIFFERENCE_S,
            )

        self.MAX_PIT_CROSSING_TIME_DIFFERENCE_MS = (
            self.MAX_PIT_CROSSING_TIME_DIFFERENCE_S * 1000
        )

        self.last_pit_crossing_timestamp_ms = None
        self.state_event_start_timestamp_ms = None
        # To prevent this set from getting too big,
        # we put a max ttl of 1 hr on any actors stored here and also allow
        # at the max 10k pit actors to be stored. These numbers are guesses
        self._crossed_pit_uuids = TTLCache(maxsize=10000, ttl=1 * 60 * 60)

        self._initialize_state_machine()

    def _initialize_state_machine(self) -> None:
        # We do not have incident state in this machine because this is an state machine for a door
        # and the door can't be in a incident (it's PITs that are really in incident).
        # Basically incident is a discreet condition that happens when certain events happen in a
        # specific order and the state machine of door is a continous running machine which is tracking
        # it's own state changes. TLDR; we check some condition every time and emit an incident if that's true
        # rather than messing the state of the state machine by unrelated/temporary actors.
        states = ["start", "open", "pit_crossed"]

        transitions = [
            {
                "trigger": "update_door_state",
                "source": "start",
                "dest": "open",
                "conditions": "_is_door_state_open",
            },
            {
                "trigger": "update_door_state",
                "source": ["open", "pit_crossed"],
                "dest": None,
                "conditions": "_is_door_state_open",
            },
            {
                "trigger": "update_door_state",
                "source": "*",
                "dest": "start",
            },
            {
                "trigger": "pit_crossing",
                "source": ["open", "pit_crossed"],
                "dest": "pit_crossed",
                "before": [
                    "_check_piggyback"
                ],  # before is called only if the conditions passes.
                "conditions": ["_is_pit_not_crossed_previously"],
            },
            # this is possible when door is partially open and the pit crosses. We don't want to handle this case yet.
            {
                "trigger": "pit_crossing",
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

        incident_list = []
        self.state_event_start_timestamp_ms = state_event_message.timestamp_ms

        if isinstance(state_event_message, State):
            self.update_door_state(door_state=state_event_message.door_state)

        if isinstance(state_event_message, Event):
            self.pit_crossing(
                subject_id=state_event_message.subject_id,
                subject_uuid=state_event_message.subject_uuid,
                incident_list=incident_list,
            )

        return incident_list[0] if len(incident_list) else None

    def _check_piggyback(
        self, subject_id, subject_uuid, incident_list, **kwargs
    ):
        last_action_timestamp_ms = self.last_pit_crossing_timestamp_ms
        if (
            last_action_timestamp_ms is not None
            and self.state_event_start_timestamp_ms
            < last_action_timestamp_ms
            + self.MAX_PIT_CROSSING_TIME_DIFFERENCE_MS
        ):
            incident_list.append(
                Incident(
                    camera_uuid=self.camera_uuid,
                    start_frame_relative_ms=self.state_event_start_timestamp_ms,
                    end_frame_relative_ms=self.state_event_start_timestamp_ms,
                    actor_ids=[subject_id],
                    track_uuid=subject_uuid,
                )
            )

    # Conditionals
    def _is_door_state_open(self, door_state, **kwargs) -> bool:
        # We are considering Partially_Open part of closed door here
        return door_state == DoorState.FULLY_OPEN

    def _is_pit_not_crossed_previously(self, subject_uuid, **kwargs) -> bool:
        """Checks that pit has not crossed prior
        Returns:
            bool: pit has not crossed
        """
        # A PIT can cross only once, this is supression logic required because bounding boxes flicker.
        return subject_uuid not in self._crossed_pit_uuids

    # Callbacks
    def on_enter_pit_crossed(self, subject_uuid: str, **kwargs) -> None:
        """Callback when pits cross
        Args:
            subject_uuid (str): subject track uuid
            **kwargs: additional arguments
        """
        self.last_pit_crossing_timestamp_ms = (
            self.state_event_start_timestamp_ms
        )
        self._crossed_pit_uuids[subject_uuid] = True

    def on_enter_start(self, **kwargs) -> None:
        self.last_pit_crossing_timestamp_ms = None
        self._crossed_pit_uuids.clear()


class PiggybackMachine(BaseStateMachine):

    NAME = "piggyback"
    INCIDENT_TYPE_NAME = "Piggy Back"
    INCIDENT_TYPE_ID = "PIGGYBACK"
    PRE_START_BUFFER_MS = 10000
    POST_END_BUFFER_MS = 5000
    INCIDENT_VERSION = "1.3"
    INCIDENT_PRIORITY = "high"

    # Cache Configs
    MAX_CACHE_SIZE = 5  # max 5 actors in a scene
    MAX_CACHE_TIME_S = 12 * 60 * 60  # 12 hours

    PER_ACTOR_COOLDOWN_THRESHOLD_S = 0

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

        if isinstance(
            state_event_message, Event
        ) and state_event_message.event_type in [
            EventType.PIT_EXITING_DOOR,
            EventType.PIT_ENTERING_DOOR,
        ]:
            return RelevantActorInfo(
                actor_id=state_event_message.object_id,
                track_uuid=state_event_message.object_uuid,
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
