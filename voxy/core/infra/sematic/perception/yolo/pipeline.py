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

from dataclasses import dataclass

import sematic

from core.infra.sematic.perception.yolo.yolo_data_generation_pipeline import (
    yolo_data_generation_pipeline,
)
from core.infra.sematic.perception.yolo.yolo_options import YoloTrainingOptions
from core.infra.sematic.perception.yolo.yolo_training_pipeline import (
    yolo_training_pipeline,
)
from core.ml.data.generation.common.pipeline import DatasetMetaData


@dataclass
class YoloTrainingResults:
    dataset_metadata: DatasetMetaData
    training_statistics: str


@sematic.func
def make_results(
    dataset_metadata: DatasetMetaData,
    training_statistics: str,
) -> YoloTrainingResults:
    """Force dependency on dataset

    Args:
        dataset_metadata (DatasetMetaData): metadata associated to dataset
        training_statistics (str): generated training statistics

    Returns:
        YoloTrainingResults: results of training
    """
    return YoloTrainingResults(
        dataset_metadata=dataset_metadata,
        training_statistics=training_statistics,
    )


@sematic.func
def pipeline(
    logset_config: str,
    dataset_config: str,
    metaverse_environment: str,
    yolo_training_options: YoloTrainingOptions,
) -> YoloTrainingResults:
    """Generate dataset and train YOLO model

    Args:
        logset_config (str): DataCollectionLogset config path
        dataset_config (str): Dataset config path
        metaverse_environment (str): Metaverse environment
        yolo_training_options (YoloTrainingOptions): Options for YOLO training

    Returns:
        YoloTrainingResults: Results of training - dataset configs and model performance statistics.
    """
    dataset, dataset_metadata = yolo_data_generation_pipeline(
        logset_config,
        dataset_config,
        metaverse_environment=metaverse_environment,
    )
    training_statistics = yolo_training_pipeline(
        dataset,
        dataset_metadata,
        yolo_training_options,
    )

    return make_results(dataset_metadata, training_statistics)
