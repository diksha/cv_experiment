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
import os
import typing
from typing import Dict, Tuple

import sematic
from loguru import logger

from core.metaverse.api.queries import register_dataset
from core.ml.data.generation.common.pipeline import (
    DatasetMetaData,
    run_pipeline,
)
from core.structs.dataset import DataCollectionLogset
from core.structs.dataset import Dataset as VoxelDataset
from core.structs.task import Task
from core.utils.yaml_jinja import load_yaml_with_jinja


def load_dataset_config(
    config_file: str, task: Task, logset: DataCollectionLogset
) -> Dict[str, object]:
    """
    Loads the logset configuration file

    Args:
        config_file (str): the file to load the logset config
        task (Task): the task to help generate the logset
        logset (DataCollectionLogset): the logset to generate the config

    Returns:
        Dict[str, object]: the loaded config dictionary
    """
    return load_yaml_with_jinja(
        config_file, task=task.to_dict(), logset=logset.to_dict()
    )


def generate_dataset(config: dict) -> DatasetMetaData:
    """
    The main entry point for the dataset generation framework. This takes as input a
    config and a logset, generates and runs a pipeline.

    Args:
        config (dict): the config required to generate the dataset
                        (sources, transforms, readers, writers)
    Returns:
        DatasetMetaData: the generated dataset object
    """
    dataset = run_pipeline(config)
    return dataset


@sematic.func(standalone=True)
def generate_and_register_dataset(
    config: typing.Dict[str, object],
    logset: DataCollectionLogset,
    metaverse_environment: typing.Optional[str] = None,
) -> Tuple[VoxelDataset, DatasetMetaData]:
    """
    Simple convenience wrapper for generating and registering the dataset
        1. generates the dataset using the logset and the config
        2. registers the dataset in the dataset registry
    config and a logset, generates and runs a pipeline.

    Args:
        config (str): the config required to generate the dataset
                            (sources, transforms, readers, writers)
        logset (DataCollectionLogset): the logset to be used in registering
        metaverse_environment (Optional[str]): metaverse environment

    Returns:
        dataset_metadata (DatasetMetaData): all metadata related to the dataset
    """
    if metaverse_environment:
        os.environ["METAVERSE_ENVIRONMENT"] = metaverse_environment
    dataset_metadata = generate_dataset(config)
    logger.info("Generated dataset")
    dataset = register_dataset(
        config=dataset_metadata.config,
        cloud_path=dataset_metadata.cloud_path,
        logset=logset,
        dataset_format=dataset_metadata.dataset_format,
    )
    dataset.set_local_download_path(dataset_metadata.local_path)
    logger.info("Registered dataset")
    return dataset, dataset_metadata
