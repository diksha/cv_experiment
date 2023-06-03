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

from core.structs.dataset import DataloaderType


class Dataloader(ABC):
    """
    Base dataloader required for all the subsequent
    dataloaders/dataset readers

    This is used to wrap pytorch/custom "Dataset"
    objects and provide utility methods like grabbing the test
    set, validation set and train set. `get_test_set`, `get_train_set`
    and similar methods are proposed to provide pytorch DataLoaders
    that are batched and ready to train. For non pytorch models
    this is upto the user discretion but the hope is to make
    training and testing as simple as possible
    """

    @abstractmethod
    def get_test_set(self):
        """
        Gets the test set dataloader
        """

    @abstractmethod
    def get_training_set(self):
        """
        Gets the train set dataloader
        """

    @abstractmethod
    def get_full_training_set(self):
        """
        Gets the all the data that is not labeled test

        (validation + training)
        """

    @abstractmethod
    def get_validation_set(self):
        """
        Gets the validation set dataloader
        """

    @classmethod
    @abstractmethod
    def dataloader_type(cls) -> DataloaderType:
        """
        Returns the destination dataloader type

        Returns:
            DataloaderType:
        """
