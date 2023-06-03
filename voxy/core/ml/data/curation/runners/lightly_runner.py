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

from core.ml.data.curation.lib.lightly_worker import LightlyWorker
from core.utils.yaml_jinja import load_yaml_with_jinja


def parse_args() -> argparse.Namespace:
    """Parses arguments

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_directory", "-i", type=str, help="s3 input directory"
    )
    parser.add_argument(
        "--output_directory", "-o", type=str, help="s3 output directory"
    )
    parser.add_argument(
        "--config_path", "-c", type=str, help="config path for lightly"
    )
    parser.add_argument(
        "--dataset_type", "-t", type=str, help="type of dataset"
    )
    parser.add_argument(
        "--dataset_name", "-n", type=str, help="name of dataset"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with LightlyWorker(
        args.dataset_name,
        args.input_directory,
        args.output_directory,
        load_yaml_with_jinja(args.config_path),
        args.dataset_type,
        notify=True,
    ) as lightly_worker:
        lightly_worker.run()
