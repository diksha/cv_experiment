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

from IPython import start_ipython


def update_argv():
    """Update sys.argv with custom configuration values."""
    sys.path.append(os.environ["BUILD_WORKSPACE_DIRECTORY"])
    # Allow import of python generate proto py files in ipython as they get generated as bin files.
    sys.path.append(
        os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], "bazel-bin")
    )


if __name__ == "__main__":
    update_argv()
    sys.exit(start_ipython())
