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
from sortedcontainers import SortedDict


class BoundedSortedDict(SortedDict):
    """Sorted dict bound to a max key length.

    This is not full proof extension and can be misused.
    For now it works with the setitem.
    """

    def __init__(self, max_length):
        self._max_len = max_length
        SortedDict.__init__(self)
        self._check_length()

    def __setitem__(self, key, value):
        SortedDict.__setitem__(self, key, value)
        self._check_length()

    def _check_length(self):
        if self._max_len is not None:
            while len(self) > self._max_len:
                self.popitem(index=0)
