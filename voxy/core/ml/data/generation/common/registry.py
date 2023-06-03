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
from core.ml.data.generation.common.logset_generator import LogsetGenerator
from core.ml.data.generation.common.stream import Stream
from core.ml.data.generation.common.writer import Writer


class StreamRegistry(BaseClassRegistry):
    """
    Registry to keep track of all streams in the dataset generation framework
    """

    REGISTRY: typing.Dict[str, typing.Type[Stream]] = {}

    @classmethod
    def get_registry(cls) -> dict:
        """
        Gets the registry for streams

        Returns:
            dict: the base static registry for streams
        """
        return cls.REGISTRY


class LogsetGeneratorRegistry(BaseClassRegistry):
    """
    Registry to keep track of all logset generators
    """

    REGISTRY: typing.Dict[str, typing.Type[LogsetGenerator]] = {}

    @classmethod
    def get_registry(cls) -> dict:
        """
        Gets the registry for logset generators

        Returns:
            dict: the base static registry for logset generators
        """
        return cls.REGISTRY


class WriterRegistry(BaseClassRegistry):
    """
    Registry to keep track of all writers in the dataset generation framework

    Writers are responsible to take in a labeled dataset collection and aggregate
    related ones to form a completed dataset.
    """

    REGISTRY: typing.Dict[str, typing.Type[Writer]] = {}

    @classmethod
    def get_registry(cls) -> dict:
        """
        Gets the registry for writers

        Returns:
            dict: the base static registry for writers
        """
        return cls.REGISTRY
