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

from core.ml.data.validation.lib.aggregate_yolo_validation import (
    generate_validation_csv,
)


def parse_args() -> argparse.Namespace:
    """
    Parse Arguments

    Returns:
        argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bucket",
        type=str,
        required=True,
        help="Bucket to search for validation results of YOLO",
    )
    parser.add_argument(
        "-r",
        "--relative_path",
        type=str,
        required=True,
        help="Relative path to search for validation results of YOLO",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_validation_csv(args.bucket, args.relative_path)
