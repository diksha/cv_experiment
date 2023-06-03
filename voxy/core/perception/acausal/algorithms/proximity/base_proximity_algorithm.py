#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.

# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

from abc import ABC, abstractmethod

from core.structs.vignette import Vignette


class BaseProximityAlgorithm(ABC):
    @abstractmethod
    def find_proximity(self, vignette: Vignette) -> dict:
        """Find the proximity for two actor classes in a given frame

        Args:
            vignette(Vignette): Frame vignette of actors

        Raises:
            NotImplementedError: Should be implemented in corresponding class

        Returns:
            dict: actor proximity map
        """
        raise NotImplementedError("Converter must implement convert method")
