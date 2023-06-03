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

import argparse
import os
import random

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from clearml import Task
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

import core.ml.algorithms.image_segmentation.data.registry as dataset_registry
import core.ml.algorithms.image_segmentation.model.registry as model_registry
import core.ml.algorithms.image_segmentation.transforms.registry as transform_registry

IMG_DIM_LANDSCPAE = (1088, 608)
IMG_DIM_PORTRAIT = (608, 1088)
DICT_TYPE_COUNT = {
    "real/positive": 0,
    "synthetic/positive": 1,
    "real/negative": 1,
}


class TrainSmp:
    def __init__(self, config):
        """Train smp model construction

        Args:
            config (_type_): training config file
        """
        self.config = config
        self.device = torch.device("cuda")
        self.n_cpu = os.cpu_count()

    def final_img_names(self, img_names: dict) -> dict:
        """Get final image names

        Args:
            img_names (dict): synthetic and real dict

        Returns:
            dict: final image names and size
        """
        result_img_info = {}
        for dataset_type, data in img_names.items():
            for size, names in data.items():
                sample_num = int(
                    self.config[f"{dataset_type}_sample_ratio"] * len(names)
                )
                if size not in result_img_info:
                    result_img_info[size] = random.sample(names, sample_num)
                else:
                    result_img_info[size].extend(
                        random.sample(names, sample_num)
                    )
        return result_img_info

    def get_imgs_info(self, local_path: str, train_dirs: list) -> dict:
        """Get image names and their corresponding size

        Args:
            local_path (str): path to the all images
            train_dirs (list): the directories that the training uses

        Returns:
            dict: dictionary of imagesize to imagenames
        """
        img_names = {"real": {}, "synthetic": {}}
        for train_dir in train_dirs:
            train_root = f"{local_path}/{train_dir}"
            for path, _, files in os.walk(train_root):

                for i, name in enumerate(files):
                    dataset_type = os.path.join(path, name).split("/")[-3]
                    if i == 0:
                        img_shape = cv2.imread(os.path.join(path, name)).shape
                        if img_shape[0] > img_shape[1]:
                            img_shape = (1080, 610, 3)
                        else:
                            img_shape = (610, 1080, 3)
                    if (
                        img_shape[0],
                        img_shape[1],
                    ) not in img_names[dataset_type]:
                        img_names[dataset_type][
                            (img_shape[0], img_shape[1])
                        ] = [os.path.join(path, name)]
                    else:
                        img_names[dataset_type][
                            (img_shape[0], img_shape[1])
                        ].append(os.path.join(path, name))

        result_img_info = self.final_img_names(img_names)

        for size, names in result_img_info.items():
            logger.info(f"size: {size}, nubmer of images: {len(names)}")
        return result_img_info

    def create_dataset(self, img_names, img_size) -> tuple:
        """Create Dataset

        Args:
            img_names (_type_): image names ot create the train and val pytorch datasets
            img_size (_type_): size of images

        Returns:
            tuple: train and validation pytorch datasets and train_weighted_sampler
        """
        if img_size[0] > img_size[1]:
            img_size = IMG_DIM_PORTRAIT
        else:
            img_size = IMG_DIM_LANDSCPAE

        img_names = np.array(img_names)
        indices = np.arange(len(img_names))

        train_indices, valid_indices = train_test_split(
            indices, test_size=0.2, random_state=42, shuffle=True
        )
        train_img_names = img_names[train_indices].tolist()
        valid_img_names = img_names[valid_indices].tolist()

        if self.config["is_augmented"]:
            albumentationstransforms = transform_registry.get_transform(
                self.config["augmentation"]
            )
            train_transforms = albumentationstransforms.compose(
                [
                    albumentationstransforms.color_transforms(),
                    albumentationstransforms.block_shuffle(),
                    albumentationstransforms.image_texture(),
                ]
            )
        else:
            train_transforms = None
        train_weighted_sampler = None
        real_positive_count = len(
            [i for i, v in enumerate(train_img_names) if "/real/positive" in v]
        )
        data_set_count = [
            real_positive_count,
            len(train_img_names) - real_positive_count,
        ]
        if all(data_set_count):
            data_set_count = np.array(data_set_count)
            weight = 1.0 / (data_set_count)

            samples_weight = np.array(
                [
                    weight[DICT_TYPE_COUNT[("/").join(name.split("/")[-3:-1])]]
                    for name in train_img_names
                ]
            )
            samples_weight = torch.from_numpy(samples_weight).double()
            train_weighted_sampler = WeightedRandomSampler(
                samples_weight, len(samples_weight)
            )

        spilldataset = dataset_registry.get_dataset(
            self.config["dataset_type"]
        )
        train = spilldataset(
            names=train_img_names,
            transform=train_transforms,
            size=img_size,
        )
        val = spilldataset(
            names=valid_img_names,
            transform=None,
            size=img_size,
        )

        return train, val, train_weighted_sampler

    def create_dataloaders(self, img_names, img_size) -> dict:
        """Create dataloaders
        Args:
            img_names (_type_): image_names
            img_size (_type_): image sizes

        Returns:
            dict: dict of train and val dataloaders
        """
        train, val, train_weighted_sampler = self.create_dataset(
            img_names, img_size
        )
        shuffle = train_weighted_sampler is None
        dataloaders = {
            "train": DataLoader(
                train,
                batch_size=self.config["batch_size"],
                shuffle=shuffle,
                num_workers=self.n_cpu,
                pin_memory=True,
                sampler=train_weighted_sampler,
            ),
            "val": DataLoader(
                val,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.n_cpu,
                pin_memory=True,
            ),
        }
        return dataloaders

    def train(self):
        """Pytorch Lightening Training
        Returns:
            _type_: saved model local directory
        """

        imgs_info = self.get_imgs_info(
            self.config["data_path"], self.config["train"]
        )

        data_loaders = []
        for img_size, image_names in imgs_info.items():
            data_loaders.append(self.create_dataloaders(image_names, img_size))
        Task.init(project_name="spill segmentation")

        smp_model = model_registry.get_model(self.config["model"])
        model = smp_model(
            "Unet",
            encoder_name="resnet18",
            in_channels=3,
            out_classes=1,
            data_loaders=data_loaders,
            model_checkpoints_s3_relative=self.config[
                "model_checkpoints_s3_relative"
            ],
            model_local_checkpoints_dir=self.config["checkpoint_dir"],
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config["checkpoint_dir"],
            filename="best",
            save_last=True,
        )
        last_check_point_dir = (
            f'{self.config["checkpoint_dir"]}/last.ckpt'
            if os.path.exists(f'{self.config["checkpoint_dir"]}/last.ckpt')
            else None
        )
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=self.config["n_epochs"],
            multiple_trainloader_mode="max_size_cycle",
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=last_check_point_dir,
        )

        trainer.fit(model)
        model_save_path = (
            f'{self.config["model_path"]}/smp-{self.config["model_name"]}.pth'
        )
        torch.save(
            trainer.model.model.state_dict(),
            model_save_path,
        )
        return model_save_path


def parse_args():
    """Get parser arguments

    Returns:
        args: user defined arguments passed in
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to training config",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of model",
        required=True,
    )
    parser.add_argument("--is_augmented", action="store_true")
    return parser.parse_args()
