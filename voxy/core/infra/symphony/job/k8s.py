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

import hashlib
import os
import re
import signal
import sys
import time

from core.infra.symphony.job.abstract import AbstractJob
from core.infra.symphony.utils.k8s_client import K8sClient


class K8sJob(AbstractJob):

    bazel_target_name = "k8s"
    job_type = "k8s"

    def __init__(self, job_struct):
        super().__init__(job_struct)
        gke_cluster_full_name = (
            "projects/sodium-carving-227300/locations/us-west1/clusters/west1"
        )
        self._k8s_client = K8sClient(
            gke_cluster_full_name=gke_cluster_full_name
        )

    def execute(self):
        jobgroup = self._job_struct.name

        def handle_cancel_signal(signum, frame):
            try:
                print(f"Deleting Job: {jobgroup}")
                self._k8s_client.delete_jobs(
                    namespace="default", jobgroup=jobgroup
                )
                print(f"{jobgroup} deleted")
            finally:
                sys.exit(1)

        signal.signal(signal.SIGTERM, handle_cancel_signal)
        signal.signal(signal.SIGQUIT, handle_cancel_signal)
        signal.signal(signal.SIGINT, handle_cancel_signal)

        status = self._k8s_client.get_jobgroup_status(
            namespace="default", jobgroup=jobgroup
        )
        # Then job already exists, don't try to relaunch.
        if status.get("total"):
            return self._check_status(jobgroup)

        items = self._job_struct.get_items()
        command = re.sub(" +", " ", self._job_struct.command)
        if items is None:
            name_and_commands_map = {self._job_struct.name: command}
        elif items:
            name_and_commands_map = {
                f"{self._job_struct.name}-{hashlib.sha256(re.sub(' +', ' ', item).encode('utf-8')).hexdigest()[:16]}": f"{command} {re.sub(' +', ' ', item)}"
                for item in items
            }
        else:
            return 0

        self._job_struct.env.update(
            {
                "SYMPHONY_CONFIG_FIRESTORE_UUID": os.getenv(
                    "SYMPHONY_CONFIG_FIRESTORE_UUID"
                )
            }
        )
        self._k8s_client.launch_jobs(
            jobgroup=jobgroup,
            container_image=self._job_struct.image,
            name_and_commands_map=name_and_commands_map,
            cpus=self._job_struct.cpus,
            memory=f"{self._job_struct.memory_mb}M",
            gpus=self._job_struct.gpus,
            env_vars=self._job_struct.env,
            container_name="runner",
            namespace="default",
            service_account_name="k8",
            gke_node_pool_name=self._job_struct.node_pool_name,
            restart_policy="OnFailure",
            attempts=self._job_struct.attempts,
            host_ipc=self._job_struct.host_ipc,
            active_deadline_seconds=self._job_struct.active_deadline_seconds,
        )
        return self._check_status(jobgroup)

    def _check_status(self, jobgroup):
        # TODO: Remove while loop and use kubernetes watch stream
        # to get job status changes rather than polling.
        while True:
            time.sleep(60)
            status = self._k8s_client.get_jobgroup_status(
                namespace="default", jobgroup=jobgroup
            )
            print("Active:", status["active"])
            print("Failed:", status["failed"])
            print("Succeeded:", status["succeeded"])
            print("Total:", status["total"])

            if status["active"] == 0:
                print("---- Succeeded Items ----")
                for item in status["succeeded_commands"]:
                    print("\t", item[-1] if isinstance(item, list) else item)
                print("---- Failed Items ----")
                for item in status["failed_commands"]:
                    print("\t", item[-1] if isinstance(item, list) else item)
                print("---- Error Logs ----")
                for log in status["error_logs"]:
                    print("\t", log[-1] if isinstance(log, list) else log)
                return 0 if status["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(K8sJob(K8sJob.parse_args_and_get_job_struct()).execute())
