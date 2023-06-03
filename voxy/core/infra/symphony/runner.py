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
import sys

from core.infra.symphony.client import SymphonyClient


def main(config_path, branch, commitsha):
    client = SymphonyClient.from_config_path(config_path)
    client.execute(branch, commitsha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--branch", type=str, default="main")
    parser.add_argument("--commitsha", type=str, default="HEAD")
    args, _ = parser.parse_known_args()
    sys.exit(main(args.config_path, args.branch, args.commitsha))
