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

import argparse
import os

from core.infra.sematic.perception.yolo.yolo_data_generation_pipeline import (
    yolo_data_generation_pipeline,
)
from core.infra.sematic.shared.utils import (
    SematicOptions,
    resolve_sematic_future,
)


def main(
    logset_config: str,
    dataset_config: str,
    sematic_options: SematicOptions,
) -> None:
    """Launch the pipeline

    Args:
        logset_config (str): See command line help
        dataset_config (str): See command line help
        sematic_options (SematicOptions): options for sematic resolvers
    """
    metaverse_environment = os.environ["METAVERSE_ENVIRONMENT"]

    # https://github.com/sematic-ai/sematic/issues/555
    # trunk-ignore(pylint/E1101)
    future = yolo_data_generation_pipeline(
        logset_config=logset_config,
        dataset_config=dataset_config,
        metaverse_environment=metaverse_environment,
    ).set(
        name="YOLO data generation pipeline",
    )
    resolve_sematic_future(future, sematic_options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--logset_config",
        type=str,
        required=True,
        help="Data collection logset config path",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Dataset config path",
    )
    SematicOptions.add_to_parser(parser)
    args = parser.parse_args()
    main(
        args.logset_config,
        args.dataset_config,
        SematicOptions.from_args(args),
    )
