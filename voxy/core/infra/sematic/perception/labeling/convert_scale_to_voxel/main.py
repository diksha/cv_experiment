#
# Copyright 2020-2021 Voxel Labs, Inc.
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
from datetime import datetime
from typing import Optional

from core.infra.sematic.perception.labeling.convert_scale_to_voxel.pipeline import (
    pipeline,
)
from core.infra.sematic.shared.utils import (
    PipelineSetup,
    SematicOptions,
    resolve_sematic_future,
)


def main(
    completion_before_date: str,
    lookback_days: float,
    converter_type: str,
    consumable_labels_fn: Optional[str],
    project_name: str,
    credentials_arn: str,
    sematic_options: SematicOptions,
):
    """Convert scale to voxel and add to metaverse

    Args:
        completion_before_date (str): date tasks must be completed before in order
            to be converted, format %Y-%m-%d %H:%M:%S
        lookback_days (float): number of days to look back from completion_before_date
            to convert tasks
        converter_type (str): Converter registered in the ScaleLabelConverterRegistry
        consumable_labels_fn (str): key for conversion function for scale to voxel conversion
        project_name (str): project name in scale to search for tasks
        credentials_arn (str): aws arn containing scale credentials
        sematic_options (SematicOptions): options for sematic pipeline
    """
    # trunk-ignore(pylint/E1101)
    future = pipeline(
        completion_before_date=completion_before_date,
        lookback_days=lookback_days,
        converter_type=converter_type,
        consumable_labels_fn=consumable_labels_fn,
        project_name=project_name,
        metaverse_environment=os.environ.get(
            "METAVERSE_ENVIRONMENT", "INTERNAL"
        ),
        credentials_arn=credentials_arn,
        pipeline_setup=PipelineSetup(),
    ).set(
        name=f"Convert scale to voxel '{project_name}'",
        tags=["P1", f"scale-to-voxel:{project_name}"],
    )
    resolve_sematic_future(future, sematic_options)


def parse_args() -> argparse.Namespace:
    """Parses arguments for the binary

    Returns:
        argparse.Namespace: Command line arguments passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--completion_before_date",
        type=str,
        default=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )
    parser.add_argument(
        "--lookback_days",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "-t",
        "--converter_type",
        type=str,
        default="VideoPlaybackAnnotationConverter",
        help="Type of task converter",
    )
    parser.add_argument(
        "-c",
        "--consumable_labels_fn",
        type=str,
        default=None,
        help="Consumable labels function",
    )
    parser.add_argument(
        "-p",
        "--project_name",
        type=str,
        default="video_playback_annotation",
        help="Name of the project",
    )
    parser.add_argument(
        "-a",
        "--credentials_arn",
        type=str,
        default=(
            "arn:aws:secretsmanager:us-west-2:203670452561:"
            "secret:scale_credentials-WHUbar"
        ),
        help="Credetials arn",
    )
    SematicOptions.add_to_parser(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(
        main(
            args.completion_before_date,
            args.lookback_days,
            args.converter_type,
            args.consumable_labels_fn,
            args.project_name,
            args.credentials_arn,
            SematicOptions.from_args(args),
        )
    )
