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
from typing import Tuple

import sematic

from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.metaverse.api.queries import get_or_create_task_and_service
from core.ml.data.generation.common.pipeline import DatasetMetaData
from core.ml.data.generation.resources.api.dataset_generator import (
    generate_and_register_dataset,
    load_dataset_config,
)
from core.ml.data.generation.resources.api.logset_generator import (
    generate_logset,
    load_logset_config,
)
from core.structs.dataset import Dataset as VoxelDataset
from core.structs.model import ModelCategory
from core.structs.task import Task, TaskPurpose


@sematic.func(
    resource_requirements=CPU_1CORE_4GB,
    standalone=True,
)
def yolo_data_generation_pipeline(
    logset_config: str,
    dataset_config: str,
    metaverse_environment: str,
) -> Tuple[VoxelDataset, DatasetMetaData]:
    """Pipeline for YOLO training

    Args:
        logset_config (str): DataCollectionLogset config path
        dataset_config (str): Dataset config path
        metaverse_environment (str): Metaverse environment to use

    Returns:
        Tuple[VoxelDataset, DatasetMetaData]: Dataset and associated MetaData for
            training
    """
    os.environ["METAVERSE_ENVIRONMENT"] = metaverse_environment
    task: Task = get_or_create_task_and_service(
        TaskPurpose["OBJECT_DETECTION_2D"],
        ModelCategory["OBJECT_DETECTION"],
        [],
    )
    logset = generate_logset(
        load_logset_config(logset_config, task=task),
    )
    (dataset, dataset_metadata) = generate_and_register_dataset(
        load_dataset_config(dataset_config, logset=logset, task=task),
        logset,
        metaverse_environment=metaverse_environment,
    )
    return dataset, dataset_metadata
