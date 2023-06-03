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
import os
import typing

import numpy as np
import pandas as pd
import PIL
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import WeightedRandomSampler

from core.ml.data.loaders.common.dataloader import (
    Dataloader as VoxelDataloader,
)
from core.ml.data.loaders.common.registry import DataloaderRegistry
from core.structs.dataset import DataloaderType

# It looks like using torch.utils.data.DataLoader kicks this off:
# trunk-ignore-all(semgrep/trailofbits.python.automatic-memory-pinning.automatic-memory-pinning)


class ImagefromCSV(TorchDataset):
    """
    Basic pytorch dataloader for grabbing a dataset from csv and a directory of images
    """

    def __init__(
        self,
        images_folder: str,
        split: str = "train",
        transforms: typing.Optional[typing.Callable] = None,
        use_weighted_random_sampler: bool = True,
        **_kwargs,
    ):
        """
        Basic initializer for loading in the image dataset

        Args:
            images_folder (str): the image folder for the dataset.
                                 Should have a labels.csv in the same location
            split (str, optional): The current split to look at. Defaults to "train".
            transforms (typing.Optional[typing.Callable], optional): the current
                                          transforms to apply. Defaults to None.
        """

        self.use_weighted_random_sampler = use_weighted_random_sampler
        self.data_frame = pd.read_csv(
            f"{images_folder}/labels.csv", header=None
        )
        # This shuffles the entire df without creating an additional old index column
        self.data_frame = self.data_frame.sample(frac=1).reset_index(drop=True)
        self.class2index = sorted(self.data_frame.iloc[:, 1].unique())
        self.label_to_id = {
            item: i for i, item in enumerate(list(self.class2index))
        }
        self.data_frame = self.data_frame.loc[
            self.data_frame.iloc[:, 2] == split
        ]
        self.images_folder = images_folder
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Returns the current length of the dataset

        Returns:
            int: the length of the dataset
        """
        return self.data_frame.shape[0]

    def __getitem__(self, index: int) -> tuple:
        """
        Grabs the data and label at the specified index

        Args:
            index (int): the current index to check

        Returns:
            tuple: the data (x) and label (y) as pytorch tensors
        """
        filename = self.data_frame.iloc[index, 0]

        label = self.class2index.index(self.data_frame.iloc[index, 1])
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label)

    def get_label_to_id(self) -> dict:
        """
        Returns dictionary of label name to id. For example:
        {
            "hard_hat" : 0,
            "no_hard_hat" : 1,
        }

        Returns:
            dict: the label to id dict
        """
        return self.label_to_id

    def get_id_to_label(self) -> dict:
        """
        Returns dictionary of id to label name. For example:
        {
            0: "hard_hat",
            1: "no_hard_hat",
        }

        Returns:
            dict: the id to label dict
        """
        return {value: key for key, value in self.label_to_id.items()}

    def get_num_labels(self) -> int:
        """
        Returns the total number of labels

        Returns:
            int: the total length of class by index
        """
        return len(self.class2index)

    def get_label_ids(self) -> list:
        """
        Returns the list of label ids. For example
        if there are 3 unique labels then it would return
        [0, 1, 2]

        Returns:
            list: the full list of label ids
        """
        return list(range(self.get_num_labels()))


class AugmentationDataset(ImagefromCSV):
    """
    Augments the (train) dataset with the compose transforms
    """

    # trunk-ignore(pylint/W0231)
    def __init__(
        self,
        dataset,
        augmentation_transforms: typing.Optional[typing.Callable] = None,
        **_kwargs,
    ):
        """Takes the train ImagefromCSV(TorchDataset) and applies transforms on it.

        Args:
            dataset (TorchDataset): ImagefromCSV(TorchDataset)
            augmentation_transforms (typing.Optional[typing.Callable], optional): Augmentations
            for train data augmentation. Defaults to None.
        """
        self.dataset = dataset
        self.augmentation_transforms = augmentation_transforms

    def __getitem__(self, index: int) -> tuple:
        """
        Returns a single image with the transforms run on it and its label
        Args:
            index(int): Image index
        Returns:
            image,label (tuple): Returns a transformed image and label pair
        """
        image = self.dataset[index][0]
        if self.augmentation_transforms:
            image = self.augmentation_transforms(image)
        label = self.dataset[index][1]
        return image, label

    def __len__(self) -> int:
        """
        Get the length of the dataset
        Returns:
            (int): length of the dataset
        """
        return len(self.dataset)


@DataloaderRegistry.register()
class ImageFromCSVDataloader(VoxelDataloader):
    """
    The top level dataloader used to organize train and test
    """

    DEFAULT_TRAIN_DATALOADER_KWARGS = {
        "batch_size": 32,
        "shuffle": False,  # This is False as weightedrandomsampler does shuffling
        "num_workers": 4,
    }
    DEFAULT_TEST_DATALOADER_KWARGS = {
        "batch_size": 32,
        "shuffle": False,
        "num_workers": 4,
    }

    def __init__(
        self,
        directory: str,
        validation_split=0.2,
        **kwargs,
    ):
        """
        Initializes the dataloader

        Args:
            directory (str): the directory with the full dataframe of train and test data
        """
        self.full_train_dataset = ImagefromCSV(
            images_folder=directory,
            split="train",
            **kwargs,
        )
        self.test_dataset = ImagefromCSV(
            images_folder=directory,
            split="test",
            **kwargs,
        )

        # do train val split
        # The training manager controls the random initialization state
        # so this is always reproducible
        train_size = int((1 - validation_split) * len(self.full_train_dataset))
        val_size = len(self.full_train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.full_train_dataset, [train_size, val_size]
        )

        # Augmentations with just the train dataset if specified in the model configs
        if "augmentation_transforms" in kwargs:
            self.train_dataset = AugmentationDataset(
                self.train_dataset,
                **kwargs,
            )

    def get_training_set(self) -> torch.utils.data.DataLoader:
        """
        Returns the batched training dataloader

        Returns:
            torch.utils.data.Dataloader: the training dataloader
        """

        if self.full_train_dataset.use_weighted_random_sampler:
            train_class_counts = np.bincount(
                [y for _, (_, y) in enumerate(self.train_dataset)]
            )
            train_label_weights = 1.0 / train_class_counts
            train_class_weights = train_label_weights[
                [y for _, (_, y) in enumerate(self.train_dataset)]
            ]

            weighted_sampler_train = WeightedRandomSampler(
                weights=train_class_weights,
                num_samples=len(self.train_dataset),
                replacement=False,
            )

            return torch.utils.data.DataLoader(
                self.train_dataset,
                sampler=weighted_sampler_train,
                pin_memory=True,
                **self.DEFAULT_TRAIN_DATALOADER_KWARGS,
            )
        self.DEFAULT_TRAIN_DATALOADER_KWARGS["shuffle"] = True
        return torch.utils.data.DataLoader(
            self.train_dataset,
            pin_memory=True,
            **self.DEFAULT_TRAIN_DATALOADER_KWARGS,
        )

    def get_full_training_set(self) -> torch.utils.data.DataLoader:
        """
        Returns the full training set (train + val split)

        Returns:
            torch.utils.data.Dataloader: the training dataloader
        """
        return torch.utils.data.DataLoader(
            self.full_train_dataset,
            pin_memory=True,
            **self.DEFAULT_TRAIN_DATALOADER_KWARGS,
        )

    def get_test_set(self) -> torch.utils.data.DataLoader:
        """
        Returns the batched test dataloader

        Returns:
            torch.utils.data.Dataloader: the test dataloader
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            pin_memory=True,
            **self.DEFAULT_TEST_DATALOADER_KWARGS,
        )

    def get_validation_set(self) -> torch.utils.data.DataLoader:
        """
        Returns the batched validation dataloader

        Returns:
            torch.utils.data.Dataloader: the validation dataloader
        """
        return torch.utils.data.DataLoader(
            self.val_dataset, **self.DEFAULT_TEST_DATALOADER_KWARGS
        )

    @classmethod
    def dataloader_type(cls) -> DataloaderType:
        """
        Get's the dataloder type for dataset

        Returns:
            DataloaderType: the destination image dataloader format
        """
        return DataloaderType.PYTORCH_IMAGE

    @classmethod
    def get_image_label_iterator(
        cls, dataframe_dataset: ImagefromCSV
    ) -> tuple:
        """
        Generator to iterate over a dataframe dataset

        Args:
            dataframe_dataset (ImagefromCSV): the current dataset to iterate over

        Yields:
            Iterator[tuple]: The stream of current images and labels
        """
        for _, row in dataframe_dataset.data_frame.iterrows():
            yield PIL.Image.open(
                os.path.join(dataframe_dataset.images_folder, row[0])
            ), dataframe_dataset.class2index.index(row[1])

    def get_label_ids(self) -> list:
        """
        Returns the list of label ids. For example
        if there are 3 unique labels then it would return
        [0, 1, 2]

        Returns:
            list: the full list of label ids
        """
        return self.full_train_dataset.get_label_ids()
