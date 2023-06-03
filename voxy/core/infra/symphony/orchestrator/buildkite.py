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

import yaml

from core.infra.symphony.job.factory import JobFactory
from core.infra.symphony.utils.config import (
    create_jobs,
    load_jobs_from_firestore,
    upload_jobs_to_firestore,
)
from core.utils.yaml_jinja import load_yaml_with_jinja


class BuildkiteOrchestrator:
    @staticmethod
    def orchestrate():
        config_firestore_path = os.getenv("SYMPHONY_CONFIG_PATH")
        if config_firestore_path:
            buildkite_jobs = create_jobs(
                load_yaml_with_jinja(config_firestore_path), "buildkite_jobs"
            )
            os.environ[
                "SYMPHONY_CONFIG_FIRESTORE_UUID"
            ] = upload_jobs_to_firestore(
                local_jobs=[], buildkite_jobs=buildkite_jobs
            )

        config_firestore_uuid = os.environ["SYMPHONY_CONFIG_FIRESTORE_UUID"]

        jobs = load_jobs_from_firestore(config_firestore_uuid)[
            "buildkite_jobs"
        ]
        job_instances = [JobFactory.get_job(job) for job in jobs]
        firestore_path = f"symphony/{config_firestore_uuid}/buildkite_jobs"

        steps = []
        for job_idx, job in enumerate(job_instances):
            steps.append(
                BuildkiteOrchestrator._generate_step(
                    job_idx, job, firestore_path
                )
            )
        return steps

    @staticmethod
    def _generate_step(job_idx, job, firestore_path):
        job_command = " ".join(job.get_run_command(firestore_path, job_idx))
        image = "203670452561.dkr.ecr.us-west-2.amazonaws.com/voxel/ci-cd-base:ubuntu20.04_voxel_v2"
        agent = (
            "cpu-aws-16c64g-x4"
            if job.job_type == "bazel"
            else "cpu-aws-4c16g-x8"
        )
        return {
            "label": job.job_struct.name,
            "key": job.job_struct.name,
            "env": {
                "SYMPHONY_CONFIG_FIRESTORE_UUID": os.getenv(
                    "SYMPHONY_CONFIG_FIRESTORE_UUID"
                )
            },
            "command": f". .buildkite/pipeline-setup && .buildkite/trap.sh {job_command}",
            "depends_on": job.job_struct.depends_on,
            "agents": {"queue": agent},
            "retry": {"automatic": [{"exit_status": -1, "limit": 3}]},
            "allow_dependency_failure": job.job_struct.allow_dependency_failure,
            "timeout_in_minutes": job.job_struct.timeout_in_minutes,
            "soft_fail": job.job_struct.soft_fail,
            "plugins": [
                {
                    "docker#v5.3.0": {
                        "image": image,
                        "propagate-environment": True,
                        "mount-ssh-agent": True,
                        "always-pull": True,
                        "userns": "host",
                        "environment": [
                            "GOOGLE_APPLICATION_CREDENTIALS_BASE64_ENCODED",
                            "AWS_REGION",
                            "AWS_ACCESS_KEY_ID",
                            "AWS_SECRET_ACCESS_KEY",
                        ],
                    }
                }
            ],
            **job.job_struct.buildkite_kwargs,
        }


if __name__ == "__main__":
    print(
        yaml.dump(
            BuildkiteOrchestrator().orchestrate(),
            default_flow_style=False,
            sort_keys=False,
        )
    )
    sys.exit(0)
