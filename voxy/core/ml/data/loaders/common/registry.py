#
# Copyright 2020-2022 Voxel Labs, Inc.
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
from core.ml.data.loaders.common.converter import Converter
from core.ml.data.loaders.common.dataloader import (
    Dataloader as VoxelDataloader,
)
from core.structs.dataset import DataloaderType
from core.structs.dataset import Dataset as VoxelDataset


class DataloaderRegistry(BaseClassRegistry):
    """
    Registry to keep track of all dataloaders in the model training
    framework
    """

    REGISTRY: typing.Dict[str, typing.Type[VoxelDataloader]] = {}

    @classmethod
    def get_registry(cls) -> dict:
        """
        Gets the registry for streams

        Returns:
            dict: the base static registry for streams
        """
        return cls.REGISTRY


class ConverterRegistry(BaseClassRegistry):
    """
    Registry to keep track of all conversions between different
    datasets and dataloaders



    For more information on how the components interact for training please see:

    Under the Dataset Converters section for a component diagram and some sample usage
    https://docs.google.com/document/d/1PzhEGRjo3RM_2ZPOYSwsFhip9iqeywcCaKEyKYOKbkU

    """

    REGISTRY: typing.Dict[str, typing.Type[Converter]] = {}

    @classmethod
    def get_registry(cls) -> dict:
        """
        Gets the registry for streams

        Returns:
            dict: the base static registry for streams
        """
        return cls.REGISTRY

    @classmethod
    def get_dataloader(
        cls,
        dataset: VoxelDataset,
        dataloader_type: DataloaderType,
        **kwargs: dict,
    ) -> VoxelDataloader:
        """
        Converts the source format of the dataset to the destination format

        For example, if the source dataset format is `CSV_IMAGE` and the destination format is
        `PYTORCH_IMAGE`, then the registry will search for the
            `CsvImageDatasetToPytorchImageDataloader`
        converter. If this is not found, then an error is raised


        Args:
            dataset (VoxelDataset): the source dataset to convert
            dataloader_type (DataloaderType): the destination dataloader type
            kwargs (dict): the any kwargs to be passed into the converter/
                           downstream dataloader

        Returns:
            VoxelDataloader: a dataloader of the type requested
        """

        def convert_to_pascal_case(string: str, delimiter="_") -> str:
            """
            Converts a string to pascal case split by the delimiter

            example: hello-world, delimiter="-"
            -> HelloWorld

            Args:
                string (str): the input string
                delimiter (str, optional): the delimiter to split the string.
                                           Defaults to "_".`

            Returns:
                str: the formatted string
            """
            return "".join(
                [substr.capitalize() for substr in string.split(delimiter)]
            )

        dataset_format_fmt = convert_to_pascal_case(dataset.format)
        dataloader_fmt = convert_to_pascal_case(dataloader_type.name)
        instance_name = (
            f"{dataset_format_fmt}DatasetTo{dataloader_fmt}Dataloader"
        )
        converter = cls.get_class(instance_name)
        return converter.convert(dataset, **kwargs)
