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
import os
import sys

from licenseheaders import main


def update_argv():
    """Update sys.argv with custom configuration values."""
    workspace_dir = os.environ["BUILD_WORKSPACE_DIRECTORY"]
    sys.argv.append(f"--tmpl={workspace_dir}/.copyright.tmpl")
    sys.argv.append("--exclude=*/portal/api/venv/*")
    sys.argv.append("--exclude=*/node_modules/*")
    sys.argv.append("--exclude=external/*")


if __name__ == "__main__":
    update_argv()
    sys.exit(main())
