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

import itertools
import time
from functools import reduce
from typing import Union

import rx
from rx import operators as ops
from rx.subject import Subject

from core.state.generators.actor import ActorStateGenerator
from core.state.generators.door import DoorStateGenerator
from core.state.generators.motion_detection_zones import (
    MotionDetectionZoneStateGenerator,
)
from core.state.generators.no_ped_zone import NoPedZoneStateGenerator
from core.state.generators.obstruction import ObstructionStateGenerator
from core.state.generators.person import PersonStateGenerator
from core.state.generators.pit import PitStateGenerator
from core.state.generators.spill import SpillStateGenerator
from core.state.publisher import Publisher
from core.structs.event import Event
from core.structs.state import State
from core.structs.vignette import Vignette


class StateController:
    def __init__(self, config: dict, otlp_meter) -> None:
        self._config = config

        self._add_epoch_time = self._config["state"]["add_epoch_time"]
        self._epoch_ms_to_add = int(time.time_ns() / 1000000)
        self._run_uuid = self._config.get("run_uuid", None)
        self._return_messages = []
        self._enabled = self._config["state"]["enabled"]

        # Initialize based on config.
        self._generators = [
            SpillStateGenerator(config),
            ActorStateGenerator(config),
            DoorStateGenerator(config),
            PitStateGenerator(config),
            PersonStateGenerator(config),
            MotionDetectionZoneStateGenerator(config),
            NoPedZoneStateGenerator(config),
            ObstructionStateGenerator(config),
        ]

        self._batch_max_latency_seconds = self._config["state"]["publisher"][
            "batch_max_latency_seconds"
        ]
        self._retry_deadline_seconds = self._config["state"]["publisher"][
            "retry_deadline_seconds"
        ]
        if self._enabled:
            self._publisher = Publisher(
                otlp_meter=otlp_meter, **self._config["state"]["publisher"]
            )

        # Ensure there are atleast 2 frames in a buffer.
        self._number_of_frames_to_buffer = max(
            config["state"]["frames_to_buffer"], 2
        )

        # Create a rx Subject which collects items
        # and use a frame length based buffer to emit collected
        # items every number of frames. Also ensure that the
        # last frame overlaps such that its last in current buffer
        # and first in next buffer.
        # The subscribe defines when the buffer emits
        # which function will be called "on_next" on the
        # next emit from the buffer.
        self._rx_subject = Subject()
        self._rx_subject.pipe(
            ops.buffer_with_count(
                self._number_of_frames_to_buffer,
                self._number_of_frames_to_buffer - 1,
            )
        ).subscribe(
            on_next=lambda items: self._group_by_type_and_actor(  # trunk-ignore(pylint/W0108)
                items
            ),
            # Add on_error
        )

    # trunk-ignore(pylint/W9011)
    def process_vignette(self, vignette: Vignette) -> list:
        if vignette.present_timestamp_ms is not None:
            self._submit_for_processing(vignette)

        return self.return_timestamp_sorted_messages_and_reset()

    # trunk-ignore(pylint/W9011)
    def return_timestamp_sorted_messages_and_reset(self) -> list:
        ret = self._return_messages
        self._return_messages = []
        return sorted(
            ret,
            key=lambda message: message.timestamp_ms,
        )

    # trunk-ignore(pylint/C0116)
    def _submit_for_processing(self, vignette: Vignette) -> None:
        # Call the generators and feed the output into rx.
        generator_responses = []
        for generator in self._generators:
            generator_responses.append(generator.process_vignette(vignette))
        self._rx_subject.on_next(generator_responses)

    # trunk-ignore(pylint/C0116)
    def _group_by_type_and_actor(
        self, list_of_list_of_generator_response: list
    ) -> None:
        list_of_generator_response = list(
            itertools.chain.from_iterable(list_of_list_of_generator_response)
        )
        states_and_events = [
            item
            for generator_response in list_of_generator_response
            for item in [generator_response.events, generator_response.states]
        ]
        flattend_states_and_events = list(
            itertools.chain.from_iterable(states_and_events)
        )
        rx.from_list(flattend_states_and_events).pipe(
            ops.group_by(lambda x: x.grouping_key),
            ops.flat_map(lambda group: group.pipe(ops.to_list())),
        ).subscribe(
            # trunk-ignore(pylint/W0108)
            on_next=lambda results: self._get_timestamps_and_reduce_to_state_changes(
                results
            ),
            # Add on_error
        )

    def _get_timestamps_and_reduce_to_state_changes(self, items: list) -> None:
        if len(items) == 0:
            return
        items = sorted(items, key=lambda item: item.timestamp_ms)
        rx.from_list(items).pipe(
            ops.group_by(lambda item: item.differentiator),
            ops.flat_map(lambda group: group.pipe(ops.to_list())),
        ).subscribe(
            # trunk-ignore(pylint/W0108)
            on_next=lambda results: self._update_end_timestamp_and_publish_for_a_group(
                results
            ),
            # Add on_error
        )

    # trunk-ignore(pylint/C0116)
    def _update_end_timestamp_and_publish_for_a_group(
        self, items: list
    ) -> None:
        # trunk-ignore(pylint/W9011)
        def merge_timestamps(
            input_list: list, item: Union[State, Event]
        ) -> list:
            if (
                input_list
                and input_list[-1].end_timestamp_ms >= item.timestamp_ms
            ):
                input_list[-1].end_timestamp_ms = max(
                    input_list[-1].end_timestamp_ms, item.end_timestamp_ms
                )
            else:
                input_list.append(item)
            return input_list

        items = sorted(
            items,
            key=lambda item: item.timestamp_ms,
        )
        items = reduce(merge_timestamps, items, [])

        length = len(items)
        for i in range(length):
            # This is for videos where our relative time starts from 0.
            if self._add_epoch_time:
                items[i].timestamp_ms += self._epoch_ms_to_add
                items[i].end_timestamp_ms += self._epoch_ms_to_add

            if self._run_uuid:
                items[i].run_uuid = self._run_uuid

            if self._enabled:
                self._publisher.publish(items[i])
            self._return_messages.append(items[i])

    # trunk-ignore(pylint/W9011)
    # trunk-ignore(pylint/C0116)
    def finalize(self) -> list:
        self._rx_subject.on_completed()
        if self._enabled:
            self._publisher.finalize()
        # Specifically for video use case to ensure
        # that the buffer got processed.
        # Find a better way to ensure all frames
        # were processed.
        time.sleep(1)
        return self.return_timestamp_sorted_messages_and_reset()
