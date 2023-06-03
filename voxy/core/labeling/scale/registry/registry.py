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
from core.labeling.scale.lib.converters.converter_base import Converter
from core.labeling.scale.task_creation.task_creation_base import TaskCreation


class ScaleLabelConverterRegistry(BaseClassRegistry):
    """
    Registry to keep track of scale converters in our labeling pipelines
    """

    REGISTRY: typing.Dict[str, typing.Type[Converter]] = {}

    @classmethod
    def get_registry(cls) -> dict:
        """
        Gets the registry for transforms

        Returns:
            dict: the base static registry for scale label converters
        """
        return cls.REGISTRY


class ScaleTaskCreatorRegistry(BaseClassRegistry):
    """
    Registry to keep track of scale converters in our labeling pipelines
    """

    REGISTRY: typing.Dict[str, typing.Type[TaskCreation]] = {}

    @classmethod
    def get_registry(cls) -> dict:
        """
        Gets the registry for transforms

        Returns:
            dict: the base static registry for scale task creators
        """
        return cls.REGISTRY
