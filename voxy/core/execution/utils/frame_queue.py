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

import threading
from collections import deque
from typing import Optional, Tuple

import numpy

from core.utils.bounded_sorted_dict import BoundedSortedDict


class FrameQueue:
    """
    This implements a thread safe video frame queue storing pts and frame data in order with a max length
    """

    def __init__(self, max_length: int):
        self._frame_map = BoundedSortedDict(max_length=max_length)
        self._frames_to_be_processed: deque[int] = deque([], maxlen=max_length)
        self._lock = threading.Lock()

    def __len__(self):
        return self.count()

    def put(self, frame: numpy.ndarray, pts: int):
        """Inserts the frame and pts into the queue"""

        with self._lock:
            self._frame_map[pts] = frame
            self._frames_to_be_processed.append(pts)

    def get(self) -> Tuple[Optional[numpy.ndarray], int]:
        """Retrieves the oldest frame from the queue and returns it along with its pts"""

        with self._lock:
            if len(self._frames_to_be_processed) == 0:
                return None, 0
            pts = self._frames_to_be_processed.popleft()
            frame = self._frame_map.pop(pts)
            return frame, pts

    def count(self) -> int:
        """Returns a count of the number of frames in this queue"""

        with self._lock:
            return len(self._frames_to_be_processed)
