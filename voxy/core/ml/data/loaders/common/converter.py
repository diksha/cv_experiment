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

from abc import ABC, abstractmethod

from core.ml.data.loaders.common.dataloader import (
    Dataloader as VoxelDataloader,
)
from core.structs.dataset import Dataset as VoxelDataset


class Converter(ABC):
    """
    Base Converter. These are used to do conversions betweens source
    registered datasets (core.structs.dataset.DatasetFormat) and the
    destination dataloaders (core.structs.dataset.DataloaderType)
    """

    @classmethod
    @abstractmethod
    def convert(cls, dataset: VoxelDataset, **kwargs: dict) -> VoxelDataloader:
        """
        Converts the dataset to the type specified

        Args:
            dataset (VoxelDataset): the input dataset
            kwargs (dict): any downstream keyword arguments to be passed into the dataloader

        Returns:
            VoxelDataloader: the converted dataloader
        """
