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

from core.common.utils.registry import BaseClassRegistry
from core.ml.training.models.model import Model


class ModelClassRegistry(BaseClassRegistry):
    """
    Registry to keep track of all models in the training framework
    """

    REGISTRY: typing.Dict[str, typing.Type[Model]] = {}

    @classmethod
    def get_registry(cls) -> dict:
        """
        Gets the registry for transforms

        Returns:
            dict: the base static registry for models
        """
        return cls.REGISTRY
