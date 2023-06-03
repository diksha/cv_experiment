import argparse

import numpy as np
import torch
import wandb
import yaml
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.wandb import WandbCallback
from fastai.data.block import DataBlock
from fastai.data.transforms import FuncSplitter, Normalize, get_image_files
from fastai.layers import Mish
from fastai.vision.augment import aug_transforms
from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats
from fastai.vision.learner import unet_learner
from fastcore.xtras import Path

import experimental.nasha.spill_segmentation.loss.registry as loss_registry
import experimental.nasha.spill_segmentation.model.registry as model_registry
import experimental.nasha.spill_segmentation.optimizer.registry as optimizer_registry


class TrainUnet:
    """
    Training script for segmentation with fastai unet
    """

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

        self.codes = np.array(config["codes"])
        self.validation_path = config.get("validation_path", None)
        wandb.init(project="spill segmentation synthetic")

    def get_mask_path(self, path):
        """Returns the mask/annotation path corresponsinf to the filename image path

        Args:
            path (str): dataset root path

        Returns:
            function: that returns full annotation mask path
        """

        def _get_y(filename):
            """Get annotation path

            Args:
                filename (Path): training image sample Path

            Returns:
                str: full annotation mask path
            """

            return f"{path}/annotation/{filename.stem}{filename.suffix}"

        return _get_y

    def get_spill_accuracy(self, prediction, target):
        """Evaluates the accuracy of the segmentation masl

        Args:
            prediction (ndarray): predicted segmentation class mask
            target (ndarray): ground truth segmentation class mask

        Returns:
            float: mean overlap
        """
        target = target.squeeze(1)
        mask = target != self.config["codes"].index("Background")
        return (prediction.argmax(dim=1)[mask] == target[mask]).float().mean()

    def file_splitter(self, filenames):
        """Split validation data

        Args:
            filenames (str): path of txt with validation files

        Returns:
            function: A function that splits out validation set
        """
        if filenames is None:
            return None
        valid = Path(filenames).read_text().split("\n")

        def _func(filepath):
            """Checks if data sample is in validation set

            Args:
                filepath (Path): file Path

            Returns:
                bool: True/False result of checking for sample in validation set
            """
            return filepath.name in valid

        def _inner(filepath, **kwargs):
            """Runs FuncSplitter to create validation split in data

            Args:
                filepath (Path): file path

            Returns:
                function: Validation set splitting function
            """
            return FuncSplitter(_func)(filepath)

        return _inner

    def get_dataloaders(self):
        """Gets the datlaoder for spills

        Returns:
            Datablock.dataloaders: dataloader for spill data - train and validation set
        """
        spill = DataBlock(
            blocks=(ImageBlock, MaskBlock(self.codes)),
            splitter=self.file_splitter(self.validation_path),
            get_items=get_image_files,
            get_y=self.get_mask_path(self.path),
            batch_tfms=[
                *aug_transforms(size=tuple(self.config["image_size"])),
                Normalize.from_stats(*imagenet_stats),
            ],
        )
        dls = spill.dataloaders(
            f"{self.path}/img",
            bs=self.config["batch_size"],
            device=self.device,
            shuffle=True,
        )
        dls.vocab = self.codes
        return dls

    def train(self):
        """Run training"""
        optimizer = optimizer_registry.get_optimizer(self.config["optimizer"])
        dataloaders = self.get_dataloaders()
        learn = unet_learner(
            dataloaders,
            model_registry.get_model(self.config["model"]),
            metrics=self.get_spill_accuracy,
            self_attention=True,
            act_cls=Mish,
            loss_func=loss_registry.get_loss(self.config["loss"]),
            opt_func=optimizer,
        )
        learn.model.to(self.device)
        learn.fit_flat_cos(
            self.config["n_epochs"],
            slice(self.config["learning_rate"]),
            cbs=[
                SaveModelCallback(),
                EarlyStoppingCallback(
                    monitor="valid_loss", min_delta=0.001, patience=8
                ),
                WandbCallback(),
            ],
        )
        torch.save(
            learn.model.state_dict(),
            f"{self.model_path}/{self.model_name}.pth",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, encoding="UTF-8") as config_file:
        parsed_config = yaml.safe_load(config_file)
    parsed_config["model_name"] = args.model_name
    train_unet = TrainUnet(config=parsed_config)
    train_unet.train()
