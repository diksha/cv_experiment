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

from core.infra.symphony.executor.buildkite import BuildkiteExecutor
from core.infra.symphony.executor.local import LocalExecutor
from core.infra.symphony.utils.config import (
    create_jobs,
    upload_jobs_to_firestore,
)
from core.utils.yaml_jinja import load_yaml_with_jinja


class SymphonyClient:
    def __init__(self, config, config_path):
        self.local_jobs = []
        self.buildkite_jobs = []
        self._config = config
        self._config_path = config_path
        self.local_jobs = create_jobs(config, "local_jobs")
        self.buildkite_jobs = create_jobs(config, "buildkite_jobs")

        self._config_firestore_uuid = upload_jobs_to_firestore(
            local_jobs=self.local_jobs, buildkite_jobs=self.buildkite_jobs
        )

    def execute(self, branch, commitsha):
        print(f"Symphony id for the jobs is: {self._config_firestore_uuid}")
        if self.local_jobs:
            LocalExecutor.execute(self._config_firestore_uuid)
        if self.buildkite_jobs:
            BuildkiteExecutor.execute(
                self._config_firestore_uuid,
                self._config_path,
                branch,
                commitsha,
            )

    @classmethod
    def from_config_path(cls, yaml_config_path):
        return SymphonyClient(
            config=load_yaml_with_jinja(yaml_config_path),
            config_path=yaml_config_path,
        )
