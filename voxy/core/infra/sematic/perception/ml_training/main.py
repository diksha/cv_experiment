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
import sys

from core.infra.sematic.shared.utils import (
    PipelineSetup,
    SematicOptions,
    resolve_sematic_future,
)
from core.ml.common.utils import add_camera_uuid_parser_arguments
from core.ml.experiments.runners.experiment_manager import run_experiment
from core.utils.yaml_jinja import load_yaml_with_jinja


def parse_args() -> argparse.Namespace:
    """Parses arguments for the binary

    Returns:
        argparse.Namespace: Command line arguments passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config_path",
        type=str,
        help="Path of the configuration defining the task",
    )
    parser.add_argument(
        "--experimenter",
        type=str,
        help="Service account or name of the person running experiments",
        default=os.environ.get("USER", "UNKNOWN"),
    )
    parser.add_argument(
        "--notify",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to notify slack",
    )
    parser = add_camera_uuid_parser_arguments(parser)
    parser.add_argument(
        "--override_config_path",
        type=str,
        help=(
            "Path of the override configuration, "
            "this will override values in the default config by merging"
        ),
        default=None,
        required=False,
    )
    SematicOptions.add_to_parser(parser)
    return parser.parse_args()


def main(args: argparse.Namespace):
    """Entry point for training using training framework and
    experiment manager

    Args:
        args (argparse.Namespace): arguments from user
    """
    future = run_experiment(
        experiment_config_path=args.experiment_config_path,
        experimenter=args.experimenter,
        notify=args.notify,
        camera_uuids=args.camera_uuids,
        organization=args.organization,
        location=args.location,
        metaverse_environment=os.environ.get(
            "METAVERSE_ENVIRONMENT", "INTERNAL"
        ),
        override_config=load_yaml_with_jinja(args.override_config_path)
        if args.override_config_path
        else None,
        pipeline_setup=PipelineSetup(),
    ).set(name="Run training experiment", tags=["P1"])

    resolve_sematic_future(future, SematicOptions.from_args(args))


if __name__ == "__main__":
    arguments = parse_args()
    sys.exit(main(arguments))
