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
import os
import sys

from black import main


def update_argv():
    """Update sys.argv with custom configuration values."""
    workspace_dir = os.environ["BUILD_WORKSPACE_DIRECTORY"]
    sys.argv.append(f"--config={workspace_dir}/pyproject.toml")
    # Last argv value is the directory containing files to be formatted
    sys.argv.append(workspace_dir)


if __name__ == "__main__":
    update_argv()
    sys.exit(main())
