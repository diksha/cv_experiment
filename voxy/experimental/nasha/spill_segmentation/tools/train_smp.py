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

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import experimental.nasha.spill_segmentation.data.registry as dataset_registry
import experimental.nasha.spill_segmentation.model.registry as model_registry
import experimental.nasha.spill_segmentation.transforms.registry as transform_registry


class TrainSmp:
    def __init__(self, config):
        """
        Initialize configurable training parameters and model names
        Args:
            config (dict): configurable model training parameters
        """
        self.config = config
        self.device = torch.device("cuda")
        self.path = config["data_path"]
        self.model_path = config["model_path"]
        self.model_name = config["model_name"]
        self.n_cpu = os.cpu_count()
        self.wandb_logger = WandbLogger(project="spill segmentation synthetic")

    def create_dataset(self):
        """Creates a pytorch dataset

        Check that image dimensions are divisible by 32,
        encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        and we will get an error trying to concat these features

        Returns:
            tuple: a tuple of the randomly selected train and val Datasets(type)
        """
        images = np.array(
            [
                os.path.splitext(file)[0]
                for file in os.listdir(f"{self.path}/img")
                if not file.startswith(".")
            ]
        )
        indices = np.arange(len(images))

        train_indices, valid_indices = train_test_split(
            indices, test_size=0.2, random_state=42, shuffle=True
        )
        train_img_names = images[train_indices].tolist()
        valid_img_names = images[valid_indices].tolist()
        if self.config["is_augmented"]:
            albumentationstransforms = transform_registry.get_transform(
                self.config["augmentation"]
            )
            train_transforms = albumentationstransforms.compose(
                [
                    albumentationstransforms.color_transforms(),
                    albumentationstransforms.block_shuffle(),
                ]
            )
        else:
            train_transforms = None
        spilldataset = dataset_registry.get_dataset(
            self.config["dataset_type"]
        )
        train = spilldataset(
            img_dir=f"{self.path}/img/",
            mask_dir=f"{self.path}/annotation/",
            names=train_img_names,
            transform=train_transforms,
            size=tuple(self.config["image_size"]),
        )
        val = spilldataset(
            img_dir=f"{self.path}/img/",
            mask_dir=f"{self.path}/annotation/",
            names=valid_img_names,
            transform=None,
            size=tuple(self.config["image_size"]),
        )

        return train, val

    def create_dataloaders(self):
        """Create train and val dataloaders

        Returns:
            dataloader(dict): a dictionary containing training and validation Dataloaders(type)
        """
        train, val = self.create_dataset()
        dataloaders = {
            "train": DataLoader(
                train,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.n_cpu,
                pin_memory=True,
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
        """Run pytorch lightning training"""
        dataloaders = self.create_dataloaders()
        smpmodel = model_registry.get_model(self.config["model"])
        model = smpmodel(
            "Unet", encoder_name="resnet18", in_channels=3, out_classes=1
        )
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=self.config["n_epochs"],
            logger=self.wandb_logger,
        )

        trainer.fit(
            model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
        )

        torch.save(
            trainer.model.model.state_dict(),
            f'{self.config["model_path"]}/smp-{self.config["model_name"]}.pth',
        )


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


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, encoding="UTF-8") as config_file:
        parsed_config = yaml.safe_load(config_file)
    parsed_config["model_name"] = args.model_name
    parsed_config["is_augmented"] = args.is_augmented
    train_smp = TrainSmp(config=parsed_config)
    train_smp.train()
