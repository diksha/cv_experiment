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

import typing
from abc import ABC, abstractmethod


class Stream(ABC):
    @abstractmethod
    def stream(self) -> typing.Iterable:
        """
        Generates iterable stream for downstream. Should return an iterable

        Returns:
            typing.Iterable: the objects to stream
        """
