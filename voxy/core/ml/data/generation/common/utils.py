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

import subprocess  # trunk-ignore(bandit/B404)


def download_dataset_to_dir(path, output_dir):
    subprocess.run(  # trunk-ignore(bandit/B607,bandit/B603)
        ["gsutil", "-m", "cp", "-r", path, output_dir],
        capture_output=False,
        check=True,
    )
