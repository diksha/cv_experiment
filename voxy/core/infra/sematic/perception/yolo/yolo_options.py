#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from typing import Dict

TRAIN_DIR_NAME = "training"
TEST_DIR_NAME = "testing"
IMAGE_DIR_NAME = "images"
LABEL_DIR_NAME = "labels"
CONFIG_DIR_NAME = "configs"
IMAGE_EXT = "png"
LABEL_EXT = "txt"
CONFIG_EXT = "yaml"


class DataSplit(Enum):
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2


def get_yolo_dataset_name(split: DataSplit, dataset_prefix: str = "") -> str:
    """Create a dataset name for YOLO
    Args:
        split (DataSplit): training, testing, or validation split
        dataset_prefix (Optional[str]): optional string to prepend to dataset name
    Returns:
        str: name of the dataset to write into file
    """
    split_name = split.name.lower()
    return f"{dataset_prefix}{split_name}_data.{LABEL_EXT}"


def get_yolo_config_name(split: DataSplit, dataset_prefix: str = "") -> str:
    """Create a config name for YOLO
    Args:
        split (DataSplit): training, testing, or validation split
        dataset_prefix (Optional[str]): optional string to prepend to dataset name
    Returns:
        str: name of the config to write into file
    """
    split_name = split.name.lower()
    return f"{dataset_prefix}{split_name}_config.{CONFIG_EXT}"


@dataclass
class YoloTrainingOptions:
    """Options for YOLO training

    Args:
        model_output_bucket (str): Cloud storage bucket will model where be stored
        model_output_relative_path (str): Path under model_output_bucket where model will be stored
        weights_path (str): Path to model weights file
        n_epochs (int): Number of epochs to train for
        batch_size (int): Training minibatch size
        image_size (int): Image size for training
        weights_cfg (str): Weights configuration
    """

    model_output_bucket: str
    model_output_relative_path: str
    weights_path: str
    n_epochs: int
    batch_size: int
    image_size: int
    weights_cfg: str

    @staticmethod
    def from_config(config: Dict[str, object]) -> "YoloTrainingOptions":
        """Construct YoloTrainingOptions from config

        Args:
            config (Namespace): Parsed commandline arguments

        Returns:
            YoloTrainingOptions: instance described by commandine arguments
        """
        return YoloTrainingOptions(**config)

    @staticmethod
    def from_args(args: Namespace) -> "YoloTrainingOptions":
        """Construct YoloTrainingOptions from parsed commandline arguments

        Args:
            args (Namespace): Parsed commandline arguments

        Returns:
            YoloTrainingOptions: instance described by commandine arguments
        """
        return YoloTrainingOptions(
            model_output_bucket=args.model_output_bucket,
            model_output_relative_path=args.model_output_relative_path,
            weights_path=args.weights_path,
            n_epochs=args.n_epochs,
            batch_size=args.training_batch_size,
            image_size=args.image_size,
            weights_cfg=args.weights_cfg,
        )

    @staticmethod
    def add_to_parser(parser: ArgumentParser) -> None:
        """Add YOLO training arguments to parser

        Args:
            parser (ArgumentParser): Parser to add options to
        """
        parser.add_argument(
            "--model_output_bucket",
            metavar="MODEL_BUCKET",
            type=str,
            default="voxel-models",
            help="Bucket where models are stored",
        )
        parser.add_argument(
            "--weights_cfg",
            type=str,
            default="yolov5m.yaml",
            help="YOLO weights configuration",
        )
        parser.add_argument(
            "--training_batch_size",
            type=int,
            default=8,
            help="YOLO training minibatch size",
        )
        parser.add_argument(
            "--image_size",
            type=int,
            default=1280,
            help="Image size to use for training",
        )
        parser.add_argument(
            "--n_epochs",
            type=int,
            default=20,
            help="Number of epochs to train YOLO models",
        )
        parser.add_argument(
            "--model_output_relative_path",
            type=str,
            default="automated/OBJECT_DETECTION_2D/yolo",
            help="Path under MODEL_BUCKET to store model outputs",
        )
        parser.add_argument(
            "--weights_path",
            type=str,
            default="yolov5m.pt",
            help="Path for initial weights",
        )
