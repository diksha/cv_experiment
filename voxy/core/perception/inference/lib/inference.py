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


class InferenceModel(ABC):
    @abstractmethod
    def infer(self, *args, **kwargs):
        """Get predictions given model and input.

        Args:
            kwargs: unused
            args: unused

        Raises:
            NotImplementedError: Should be implemented by child class.
        """
        raise NotImplementedError(
            "The baseclass cannot infer method. \
                This should be implemented in the child class"
        )
