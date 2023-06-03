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
from datetime import datetime

from core.labeling.scale.lib.converters.scale_voxel_conversion import (
    convert_scale_to_voxel,
)


def main(args):
    convert_scale_to_voxel(
        args.completion_before_date,
        args.lookback_days,
        args.type,
        args.consumable_labels_fn,
        args.project_name,
        args.credentials_arn,
    )


def parse_args():
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
        "--type",
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
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
