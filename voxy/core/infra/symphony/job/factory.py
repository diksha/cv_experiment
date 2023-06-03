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

from core.infra.symphony.job.bazel import BazelJob
from core.infra.symphony.job.k8s import K8sJob


class JobFactory:

    job_mapping = {
        BazelJob.job_type: BazelJob,
        K8sJob.job_type: K8sJob,
    }

    @staticmethod
    def get_job(job_struct):
        if job_struct.type not in JobFactory.job_mapping:
            raise RuntimeError(f"Unknown job type: {job_struct.type}")

        return JobFactory.job_mapping[job_struct.type](job_struct)
