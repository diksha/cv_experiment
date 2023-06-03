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
import sys
import uuid

import yaml

from core.infra.sematic.perception.spill.spill_training_pipeline import (
    spill_training_pipeline,
)
from core.infra.sematic.shared.utils import (
    SematicOptions,
    resolve_sematic_future,
)


def main(
    config: dict,
    sematic_options: SematicOptions,
) -> int:
    """Function to call on training pipleline

    Args:
        config (dict): dictionary of training parameters and dataset
        sematic_options (SematicOptions): options for sematic resolver

    Returns:
        int: _description_
    """

    future = spill_training_pipeline(config_training=config).set(
        name="spill segmentation training",
        tags=[f"model:{config['model_name']}"],
    )

    resolve_sematic_future(future, sematic_options)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config",
        required=True,
        help="training config file",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=uuid.uuid4().hex,
        help="model name for labels (default is new UUID)",
    )
    SematicOptions.add_to_parser(parser)

    args, _ = parser.parse_known_args()
    with open(args.config, encoding="utf-8") as c:
        parsed_config = yaml.safe_load(c)
    parsed_config["model_name"] = args.model_name
    sys.exit(
        main(
            parsed_config,
            SematicOptions.from_args(args),
        )
    )
