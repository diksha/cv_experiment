#
# Copyright 2022-2023 Voxel Labs, Inc.
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
import json
from typing import Tuple

from core.infra.sematic.perception.yolo.yolo_options import YoloTrainingOptions
from core.infra.sematic.perception.yolo.yolo_training_pipeline import (
    yolo_training_pipeline,
)
from core.infra.sematic.shared.utils import (
    SematicOptions,
    resolve_sematic_future,
)
from core.metaverse.api.queries import get_dataset
from core.ml.data.generation.common.pipeline import DatasetMetaData
from core.structs.dataset import Dataset as VoxelDataset
from core.structs.dataset import DatasetFormat


def get_dataset_and_metadata(
    dataset_uuid: str, local_path: str
) -> Tuple[VoxelDataset, DatasetMetaData]:
    """Get the dataset and metatata from a data generation pipeline
    Args:
        dataset_uuid (str): uuid of a dataset
        local_path (str): local path writer used to generate the dataset
            MUST MATCH DATA GENERATION PIPELINE!
    Returns:
        Tuple[VoxelDataset, DatasetMetaData]: dataset, metadata pair
    """
    dataset = get_dataset(dataset_uuid)
    # trunk-ignore-begin(pylint/E1101)
    # Dataset is a child class of recursive simple namespace which
    # instantiates args from kwargs, so its attributes are not visible
    # to the linter
    return dataset, DatasetMetaData(
        cloud_path=dataset.path,
        config=json.loads(dataset.config),
        dataset_format=DatasetFormat[dataset.format],
        local_path=local_path,
    )
    # trunk-ignore-end(pylint/E1101)


def main(
    dataset_uuid: str,
    dataset_local_path: str,
    yolo_training_options: YoloTrainingOptions,
    sematic_options: SematicOptions,
) -> None:
    """Run YOLO training sematic job

    Args:
        dataset_uuid (str): uuid of dataset generated for training in metaverse
        dataset_local_path (str): local path used to generate the dataset associated
            to dataset_uuid
        yolo_training_options (YoloTrainingOptions): Options for YOLO training
        sematic_options (SematicOptions): options for sematic resolvers
    """
    dataset, dataset_metadata = get_dataset_and_metadata(
        dataset_uuid, dataset_local_path
    )
    future = yolo_training_pipeline(
        dataset,
        dataset_metadata,
        yolo_training_options,
    ).set(
        name=f"YOLO training pipeline for {dataset_uuid}",
        tags=[f"dataset:{dataset_uuid}"],
    )

    resolve_sematic_future(future, sematic_options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--dataset_uuid",
        type=str,
        required=True,
        help="Dataset uuid from metaverse",
    )
    parser.add_argument(
        "--dataset_local_path",
        type=str,
        required=True,
        help=(
            "Local path used in dataset generation pipeline."
            "MUST MATCH LOCAL PATH USED IN DATASET GENERATION!"
        ),
    )
    SematicOptions.add_to_parser(parser)
    YoloTrainingOptions.add_to_parser(parser)
    args = parser.parse_args()
    main(
        args.dataset_uuid,
        args.dataset_local_path,
        YoloTrainingOptions.from_args(args),
        SematicOptions.from_args(args),
    )
