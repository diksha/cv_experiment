#
# Copyright 2022 Voxel Labs, Inc.
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
from typing import Optional

import sematic
from loguru import logger

from core.infra.sematic.perception.yolo.yolo_options import (
    CONFIG_DIR_NAME,
    TEST_DIR_NAME,
    TRAIN_DIR_NAME,
    DataSplit,
    YoloTrainingOptions,
    get_yolo_config_name,
)
from core.infra.sematic.shared.resources import GPU_8CPU_32GB_1x
from core.ml.data.generation.common.pipeline import DatasetMetaData
from core.ml.data.validation.lib.aggregate_yolo_validation import (
    generate_validation_csv,
)
from core.structs.dataset import Dataset as VoxelDataset
from core.utils.aws_utils import (
    download_directory_from_s3,
    get_secret_from_aws_secret_manager,
    upload_directory_to_s3,
)
from core.utils.subprocess_utils import logged_subprocess_call


def prepare_data_dirs() -> None:
    """Prepare local data directories.

    This is separated so it can be easily mocked.
    """
    os.makedirs("/data/runs")
    try:
        os.rmdir("/usr/src/app/runs")
    except FileNotFoundError:
        pass
    os.symlink("/data/runs", "/usr/src/app/runs")


def setup_data(
    dataset: VoxelDataset,
    local_path: str,
    split: DataSplit,
    model_output_bucket: Optional[str] = None,
    model_output_relative_path: Optional[str] = None,
) -> str:
    """Download data to local image to prepare for YOLO training

    Args:
        dataset (VoxelDataset): Dataset from metaverse
        local_path (str): local path to download dataset (provided by DatasetMetaData)
        split (DataSplit): training, validation, split
        model_output_bucket (Optional[str]): cloud storage bucket to download
            model from (if applicable)
        model_output_relative_path (Optional[str]): cloud storage relative path in
            bucket to download model from (if applicable)

    Returns:
        str: training dataset configuration path

    Raises:
        RuntimeError: invalid split passed (only training and testing splits valid)
    """
    if split == DataSplit.TRAINING:
        split_directory = TRAIN_DIR_NAME
    elif split == DataSplit.TESTING:
        split_directory = TEST_DIR_NAME
    else:
        raise RuntimeError(f"{split.name} is not a valid split for setup_data")

    dataset.download(local_path)

    if model_output_bucket and model_output_relative_path:
        model_weights = (
            f"{model_output_relative_path}/{dataset.uuid}/"
            f"model_output/yolo_automated_{dataset.uuid}/weights"
        )
        logger.info(f"Downloading model weights from {model_weights}")
        download_directory_from_s3(
            os.path.join(local_path, "weights"),
            model_output_bucket,
            model_weights,
        )

    prepare_data_dirs()

    logger.info(f"Data setup finished for {split.name}")

    return os.path.join(
        local_path,
        split_directory,
        CONFIG_DIR_NAME,
        get_yolo_config_name(split),
    )


def setup_environment():
    """Sets up environment variables for yolo run"""
    os.environ["CLEARML_WEB_HOST"] = get_secret_from_aws_secret_manager(
        "CLEARML_WEB_HOST"
    )
    os.environ["CLEARML_API_HOST"] = get_secret_from_aws_secret_manager(
        "CLEARML_API_HOST"
    )
    os.environ["CLEARML_FILES_HOST"] = get_secret_from_aws_secret_manager(
        "CLEARML_FILES_HOST"
    )
    os.environ["CLEARML_API_ACCESS_KEY"] = get_secret_from_aws_secret_manager(
        "CLEARML_API_ACCESS_KEY"
    )
    os.environ["CLEARML_API_SECRET_KEY"] = get_secret_from_aws_secret_manager(
        "CLEARML_API_SECRET_KEY"
    )


@sematic.func(
    resource_requirements=GPU_8CPU_32GB_1x,
    base_image_tag="yolov5-base",
    standalone=True,
)
def train(
    dataset: VoxelDataset,
    dataset_metadata: DatasetMetaData,
    yolo_training_options: YoloTrainingOptions,
) -> str:
    """Train YOLO model

    Args:
        dataset (VoxelDataset): Dataset from metaverse
        dataset_metadata (DatasetMetaData): Associated metadata related to the dataset
        yolo_training_options (YoloTrainingOptions): Options for YOLO training

    Returns:
        str: path to model
    """
    setup_environment()
    training_config_path = setup_data(
        dataset,
        dataset_metadata.local_path,
        split=DataSplit.TRAINING,
    )
    # Force conda environment
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["OMP_NUM_THREADS"] = "1"

    logged_subprocess_call(
        [
            # Intentionally trying to use YOLO docker python, NOT voxel hermetic python
            "/opt/conda/bin/python",
            "train.py",
            "--weights",
            yolo_training_options.weights_path,
            "--epochs",
            str(yolo_training_options.n_epochs),
            "--batch-size",
            str(yolo_training_options.batch_size),
            "--data",
            training_config_path,
            "--img",
            str(yolo_training_options.image_size),
            "--name",
            f"yolo_automated_{dataset.uuid}",
            "--cfg",
            yolo_training_options.weights_cfg,
        ],
        cwd="/usr/src/app",
        env=env,
        checked=True,
    )

    return upload_directory_to_s3(
        yolo_training_options.model_output_bucket,
        "/data/runs/train",
        os.path.join(
            yolo_training_options.model_output_relative_path,
            dataset.uuid,
            "model_output",
        ),
    )


@sematic.func(
    resource_requirements=GPU_8CPU_32GB_1x,
    base_image_tag="yolov5-base",
    standalone=True,
)
def val(
    dataset: VoxelDataset,
    dataset_metadata: DatasetMetaData,
    yolo_training_options: YoloTrainingOptions,
    model_output: str,
) -> str:
    """Validate YOLO model

    Args:
        dataset (VoxelDataset): Dataset from metaverse
        dataset_metadata (DatasetMetaData): Associated metadata related to the dataset
        yolo_training_options (YoloTrainingOptions): options for training YOLO
        model_output (str): dummy for sematic dependency

    Returns:
        str: Path to validation output
    """
    setup_environment()
    setup_data(
        dataset,
        dataset_metadata.local_path,
        split=DataSplit.TESTING,
        model_output_bucket=yolo_training_options.model_output_bucket,
        model_output_relative_path=yolo_training_options.model_output_relative_path,
    )
    # force conda environment
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    logged_subprocess_call(
        [
            # Intentionally trying to use YOLO docker python, NOT voxel hermetic python
            "/opt/conda/bin/python",
            "val_wrapper.py",
            "--data-folder",
            f"{dataset_metadata.local_path}/{TEST_DIR_NAME}/{CONFIG_DIR_NAME}",
            "--workers",
            "0",
            "--weights",
            "/data/weights/best.pt",
            "--task",
            "test",
            "--img",
            str(yolo_training_options.image_size),
            "--iou",
            "0.65",
            "--half",
        ],
        cwd="/usr/src/app",
        env=env,
        checked=True,
    )
    return upload_directory_to_s3(
        yolo_training_options.model_output_bucket,
        "/data/runs/val",
        os.path.join(
            yolo_training_options.model_output_relative_path,
            dataset.uuid,
            "model_val",
        ),
    )


@sematic.func
def aggregate_val(
    val_output: str, model_output_bucket: str, val_path: str
) -> str:
    """Aggregate YOLO validation data

    Args:
        val_output (str): dummy for dependency
        model_output_bucket (str): model output bucket
        val_path (str): path to output validation data

    Returns:
        str: Path to validation output
    """
    return generate_validation_csv(model_output_bucket, val_path)


@sematic.func
def yolo_training_pipeline(
    dataset: VoxelDataset,
    dataset_metadata: DatasetMetaData,
    yolo_training_options: YoloTrainingOptions,
) -> str:
    """Sematic job for training YOLO models

    Args:
        dataset (VoxelDataset): Dataset from metaverse
        dataset_metadata (DatasetMetaData): Associated metadata related to the dataset
        yolo_training_options (YoloTrainingOptions): YOLO training options

    Returns:
        str: Path to generated model validation data
    """
    model_output = train(
        dataset,
        dataset_metadata,
        yolo_training_options,
    )
    val_output = val(
        dataset,
        dataset_metadata,
        yolo_training_options,
        model_output,
    )
    return aggregate_val(
        val_output,
        yolo_training_options.model_output_bucket,
        os.path.join(
            yolo_training_options.model_output_relative_path,
            dataset.uuid,
            "model_val",
        ),
    )
