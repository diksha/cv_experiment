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

from loguru import logger

from core.execution.lib.production_lib import (
    ENVIRONMENT_DEV,
    RUNTIME_ENVIRONMENTS,
    ProductionRunner,
)
from core.utils.logger import ALL_LOGGING_LEVELS, LOG_LEVEL_INFO


def parse_args() -> argparse.Namespace:
    """Parse arguments from CLI

    Returns:
        argparse.Namespace: argument values
    """
    parser = argparse.ArgumentParser(
        description="Production Graph Runner", allow_abbrev=False
    )
    parser.add_argument("--camera_config_path", type=str, required=True)
    parser.add_argument(
        "--environment",
        type=str,
        choices=RUNTIME_ENVIRONMENTS,
        default=ENVIRONMENT_DEV,
    )
    parser.add_argument(
        "--logging_level",
        type=str.upper,
        choices=ALL_LOGGING_LEVELS,
        default=LOG_LEVEL_INFO,
    )
    parser.add_argument(
        "--serialize_logs", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--triton_server_url", type=str, default="")
    _args, _ = parser.parse_known_args()
    return _args


if __name__ == "__main__":
    args = parse_args()

    logger.info("Starting process with args ")
    logger.info(args)

    try:
        ProductionRunner(
            camera_config_path=args.camera_config_path,
            env=args.environment,
            logging_level=args.logging_level,
            serialize_logs=args.serialize_logs,
            triton_server_url=args.triton_server_url,
        ).run()
    except KeyboardInterrupt:
        print("INTERRUPTED")
        try:
            sys.exit(1)
        except SystemExit:
            # trunk-ignore(pylint/W0212)
            os._exit(1)
