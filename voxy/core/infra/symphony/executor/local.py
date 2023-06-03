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

from core.infra.symphony.job.factory import JobFactory
from core.infra.symphony.orchestrator.local import LocalOrchestrator
from core.infra.symphony.utils.config import load_jobs_from_firestore


class LocalExecutor:
    @staticmethod
    def execute(config_firestore_uuid):
        jobs = load_jobs_from_firestore(config_firestore_uuid)["local_jobs"]
        job_instances = [JobFactory.get_job(job) for job in jobs]
        LocalOrchestrator.orchestrate(
            config_firestore_uuid,
            job_instances,
            firestore_path=f"symphony/{config_firestore_uuid}/local_jobs",
        )
