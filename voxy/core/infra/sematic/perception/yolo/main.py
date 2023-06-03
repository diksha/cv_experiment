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

import argparse
import os

import mergedeep

from core.infra.sematic.perception.yolo.pipeline import pipeline
from core.infra.sematic.perception.yolo.yolo_options import YoloTrainingOptions
from core.infra.sematic.shared.utils import (
    SematicOptions,
    resolve_sematic_future,
)
from core.utils.yaml_jinja import load_yaml_with_jinja


def main(
    logset_config: str,
    dataset_config: str,
    yolo_training_options: YoloTrainingOptions,
    sematic_options: SematicOptions,
    dry_run: bool,
) -> None:
    """Run YOLO training sematic job

    Args:
        logset_config (str): See command line help
        dataset_config (str): See command line help
        yolo_training_options (YoloTrainingOptions): YOLO training options
        sematic_options (SematicOptions): options for Sematic resolvers
        dry_run (bool): waits until future has finished
    """
    metaverse_environment = os.environ["METAVERSE_ENVIRONMENT"]

    # https://github.com/sematic-ai/sematic/issues/555
    # trunk-ignore(pylint/E1101)
    future = pipeline(
        logset_config,
        dataset_config,
        metaverse_environment,
        yolo_training_options,
    ).set(
        name=f"YOLO dataset generation and training pipeline{' (smoke test)' if dry_run else ''}",
        tags=["P2"],
    )
    resolve_sematic_future(
        future,
        sematic_options,
        block_run=dry_run,
    )


def get_production_config() -> dict:
    """
    Loads and returns the pipeline configs (production)

    Returns:
        dict: regular production config
    """
    return load_yaml_with_jinja(
        "core/infra/sematic/perception/yolo/configs/production.yaml"
    )


def get_smoketest_config() -> dict:
    """
    Loads and returns the pipeline configs (smoketest and regular)

    Returns:
        dict: smoke test config
    """
    return load_yaml_with_jinja(
        "core/infra/sematic/perception/yolo/configs/smoke_test.yaml"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--dry-run", action="store_true", help="Whether to run the smoke test"
    )
    SematicOptions.add_to_parser(parser)
    args = parser.parse_args()
    config_values = get_production_config()
    if args.dry_run:
        mergedeep.merge(config_values, get_smoketest_config())
    main(
        yolo_training_options=YoloTrainingOptions.from_config(
            config_values.get("yolo_training_options")
        ),
        sematic_options=SematicOptions.from_args(args),
        dry_run=args.dry_run,
        **config_values.get("dataset_generation"),
    )
