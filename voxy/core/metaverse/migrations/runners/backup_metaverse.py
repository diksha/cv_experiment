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

from core.metaverse.migrations.lib.backup_wrapper import MetaverseBackupWrapper


def parse_args() -> argparse.Namespace:
    """
    Argument Parser
    Returns:
        argparse.Namespace: cl args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--metaverse_environment",
        type=str,
        required=True,
        help="Metaverse environment to backup",
    )
    return parser.parse_args()


def backup_metaverse(metaverse_environment: str):
    """Main runner to backup metaverse
    Args:
        metaverse_environment (str): metaverse env
    """
    wrapper = MetaverseBackupWrapper(metaverse_environment)
    wrapper.backup_metaverse()


if __name__ == "__main__":
    args = parse_args()
    backup_metaverse(
        args.metaverse_environment,
    )
