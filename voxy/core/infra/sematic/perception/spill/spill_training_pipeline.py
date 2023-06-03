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
import typing

import sematic
from loguru import logger

from core.infra.sematic.shared.resources import GPU_8CPU_32GB_1x
from core.ml.algorithms.image_segmentation.sanity_check.random_check import (
    SanityChecker,
)
from core.ml.algorithms.image_segmentation.tools.train_smp import TrainSmp
from core.utils.aws_utils import (
    download_directory_from_s3,
    get_secret_from_aws_secret_manager,
    upload_file,
)

CHECKPOINT_DIR_SUFFIX = "checkpoints"


def setup_clearml():
    """Setup ClearML config parameters"""
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


def sanity_check(config_training: typing.Dict[str, object]):
    """Checks the dataset quality

    Args:
        config_training (typing.Dict[str, object]): Training config parameters
    """
    logger.info("Running sanity check on dataset ...")
    sanity_checker = SanityChecker(
        sample_fraction=0.001,
        config_training=config_training,
    )
    sanity_checker.save_training_config()
    sanity_checker.count_training_samples()
    sanity_checker.random_upload_images()


def setup_data(
    data_extraction_bucket: str,
    data_extraction_relative_path: str,
    paths_to_download: list,
    experiment_name: str,
    download_all_data: bool = True,
    local_path: str = "/data",
    model_path: str = "/model",
) -> str:
    """Dataset setup for the pipeline

    Args:
        data_extraction_bucket (str): s3 bucket of the data
        data_extraction_relative_path (str): root directory of the dataset
        paths_to_download (list): list of paths to download
        experiment_name (str): experiment name relative directory in s3
        download_all_data (bool): download all data or per path
        local_path (str, optional): local directory to store the dataset. Defaults to "/data".
        model_path (str, optional): model save local directory. Defaults to "/model".
    """

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    if not download_all_data:
        logger.info("Downloading per camera")
        for path in paths_to_download:
            path = path.replace("/img", "")
            logger.info(f"Downloading dataset from {path}")
            local_sub_path = f"{local_path}/{path}"
            s3_local_sub_path = f"{data_extraction_relative_path}/{path}"
            download_directory_from_s3(
                local_sub_path,
                data_extraction_bucket,
                s3_local_sub_path,
            )
    else:
        logger.info("Downloading All cameras")
        logger.info(
            f"Downloading dataset from {data_extraction_relative_path}"
        )
        download_directory_from_s3(
            local_path,
            data_extraction_bucket,
            data_extraction_relative_path,
        )
    check_points_dir = f"{model_path}/{CHECKPOINT_DIR_SUFFIX}/"
    if not os.path.exists(check_points_dir):
        os.makedirs(check_points_dir)

    download_directory_from_s3(
        check_points_dir,
        data_extraction_bucket,
        f"{experiment_name}/{CHECKPOINT_DIR_SUFFIX}",
    )
    logger.info("Data setup finished")


@sematic.func(
    resource_requirements=GPU_8CPU_32GB_1x,
    standalone=True,
)
def spill_training_pipeline(config_training: typing.Dict[str, object]) -> str:
    """Sematic training for spill segmentation

    Args:
        config_training (typing.Dict[str, object]): training config parameters

    Returns:
        str: saved model path
    """
    sanity_check(config_training=config_training)
    setup_clearml()
    config_training["train"] = list(set(config_training["train"]))
    config_training[
        "checkpoint_dir"
    ] = f'{config_training["model_path"]}/{CHECKPOINT_DIR_SUFFIX}'
    config_training["model_checkpoints_s3_relative"] = os.path.join(
        config_training["model_save_relative_path"],
        config_training["model_name"],
        CHECKPOINT_DIR_SUFFIX,
    )
    experiment_name = os.path.join(
        config_training["model_save_relative_path"],
        config_training["model_name"],
    )
    setup_data(
        data_extraction_bucket=config_training["data_extraction_bucket"],
        data_extraction_relative_path=config_training[
            "data_extraction_relative_path"
        ],
        paths_to_download=config_training["train"],
        experiment_name=experiment_name,
        download_all_data=config_training["download_all_data"],
        local_path=config_training["data_path"],
        model_path=config_training["model_path"],
    )
    train_smp = TrainSmp(config=config_training)
    model_save_path = train_smp.train()
    model_s3_save_path = os.path.join(
        config_training["model_save_relative_path"],
        config_training["model_name"],
        model_save_path.split("/")[-1],
    )
    logger.info(f"Saved Model:{model_s3_save_path}")
    upload_file(
        "voxel-users",
        model_save_path,
        model_s3_save_path,
    )
    return model_s3_save_path
