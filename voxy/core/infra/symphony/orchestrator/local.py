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


class LocalOrchestrator:
    @staticmethod
    def orchestrate(config_firestore_uuid, jobs, firestore_path):
        for job_idx, job in enumerate(jobs):
            subprocess.run(
                job.get_run_command(firestore_path, job_idx),
                capture_output=False,
                # Set the current working directory such that we use the
                # correct bazel irrespective of where it is run from.
                cwd=os.environ["BUILD_WORKSPACE_DIRECTORY"],
                env=dict(
                    os.environ,
                    SYMPHONY_CONFIG_FIRESTORE_UUID=config_firestore_uuid,
                ),
                check=True,
            )
