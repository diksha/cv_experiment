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


class Buffer:
    def __init__(self, max_past_frames, max_future_frames):
        self._max_past_frames = max_past_frames
        self._max_future_frames = max_future_frames
        self._buffer = []

    def process(self, frame_struct):
        self._buffer.append(frame_struct)
        result = [None, [], []]

        if len(self._buffer) > self._max_future_frames:
            current_frame_index = len(self._buffer) - (
                self._max_future_frames + 1
            )
            history_start_index = max(
                0, current_frame_index - self._max_past_frames
            )
            future_end_index = current_frame_index + self._max_future_frames
            result = [
                self._buffer[current_frame_index],
                self._buffer[history_start_index:current_frame_index],
                self._buffer[current_frame_index + 1 : future_end_index],
            ]

        if (
            len(self._buffer)
            > self._max_past_frames + 1 + self._max_future_frames
        ):
            self._buffer.pop(0)

        return result
