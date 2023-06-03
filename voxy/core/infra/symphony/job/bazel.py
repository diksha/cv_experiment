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
import subprocess
import sys

from core.infra.symphony.job.abstract import AbstractJob


class BazelJob(AbstractJob):

    bazel_target_name = "bazel"
    job_type = "bazel"

    def execute(self):
        command = " ".join(["./bazel", "run", self._job_struct.command]).split(
            " "
        )
        subprocess.run(
            command,
            capture_output=False,
            # Set the current working directory such that we use the
            # correct bazel irrespective of where it is run from.
            cwd=os.environ["BUILD_WORKSPACE_DIRECTORY"],
            env=dict(
                os.environ | self._job_struct.env,
                SYMPHONY_CONFIG_FIRESTORE_UUID=os.getenv(
                    "SYMPHONY_CONFIG_FIRESTORE_UUID", ""
                ),
            ),
            check=True,
        )


if __name__ == "__main__":
    sys.exit(BazelJob(BazelJob.parse_args_and_get_job_struct()).execute())
