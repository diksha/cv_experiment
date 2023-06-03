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

from abc import ABC, abstractmethod


class Transform(ABC):
    """
    Transform is a core object that are designed to be composed
    into a pipeline. Generating a transform object should allow
    generating arbitrary composed.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
