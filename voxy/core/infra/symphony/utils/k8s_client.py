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
import google.auth
from google.auth.transport import requests
from google.cloud.container_v1 import ClusterManagerClient
from kubernetes import client
from kubernetes.client.rest import ApiException


class K8sClient:
    def get_instance(self):
        auth_request = requests.Request()
        self.credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.credentials.refresh(auth_request)
        cluster_manager_client = ClusterManagerClient(
            credentials=self.credentials
        )
        cluster = cluster_manager_client.get_cluster(
            name=self.gke_cluster_full_name
        )
        configuration = client.Configuration()
        configuration.host = f"https://{cluster.endpoint}:443"
        configuration.verify_ssl = False
        configuration.api_key = {
            "authorization": "Bearer " + self.credentials.token
        }
        client.Configuration.set_default(configuration)
        api_instance = client.BatchV1Api()
        return api_instance

    def __init__(self, gke_cluster_full_name):
        self.gke_cluster_full_name = gke_cluster_full_name
        self.api_instance = self.get_instance()
        self.node_pool_map = {
            ("west1", 0): "preemptible-cpu-only-pool-large",
            ("west1", 1): "preemptible-gpu-pool",
            ("west1", 2): "preemptible-2-gpu-pool",
        }

    def launch_job(
        self,
        name,
        jobgroup,
        container_image,
        command,
        cpus,
        memory,
        gpus,
        env_vars,
        container_name,
        namespace,
        service_account_name,
        gke_node_pool_name,
        restart_policy,
        attempts,
        host_ipc,
        active_deadline_seconds,
    ):
        body = client.V1Job(api_version="batch/v1", kind="Job")
        body.metadata = client.V1ObjectMeta(namespace=namespace, name=name)
        body.status = client.V1JobStatus()

        env_list = []
        for env_name, env_value in env_vars.items():
            env_list.append(client.V1EnvVar(name=env_name, value=env_value))

        resources = client.V1ResourceRequirements(
            limits={
                "cpu": cpus,
                "memory": memory,
                "nvidia.com/gpu": gpus,
            }
        )

        container = client.V1Container(
            name=container_name,
            image=container_image,
            env=env_list,
            command=["/bin/sh", "-c"],
            args=[command],
            resources=resources,
        )

        template = client.V1PodTemplate()
        template.template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"jobgroup": jobgroup})
        )

        if gke_node_pool_name is None:
            gke_node_pool_name = self.node_pool_map.get(
                (self.gke_cluster_full_name.split("/")[-1], gpus), None
            )
        node_selector = None

        if gke_node_pool_name is not None:
            node_selector = {
                "cloud.google.com/gke-nodepool": gke_node_pool_name
            }

        template.template.spec = client.V1PodSpec(
            containers=[container],
            restart_policy=restart_policy if restart_policy else None,
            service_account_name=service_account_name
            if service_account_name
            else None,
            node_selector=node_selector,
            active_deadline_seconds=active_deadline_seconds,
            host_ipc=host_ipc,
        )

        # ttl_seconds_after_finished cleans up the job
        # after specified seconds, basically removes the job.
        body.spec = client.V1JobSpec(
            ttl_seconds_after_finished=86400,
            template=template.template,
            backoff_limit=attempts - 1,
            active_deadline_seconds=active_deadline_seconds,
        )

        api_response = self.api_instance.create_namespaced_job(
            namespace, body, pretty=True
        )
        return api_response

    def get_job_status(self, name, namespace):
        api_response = self.api_instance.read_namespaced_job_status(
            name, namespace
        )
        return api_response.job.status

    def launch_jobs(
        self,
        jobgroup,
        container_image,
        name_and_commands_map,
        cpus,
        memory,
        gpus,
        env_vars,
        container_name="runner",
        namespace="default",
        service_account_name="k8",
        gke_node_pool_name=None,
        restart_policy="OnFailure",
        attempts=1,
        host_ipc=False,
        active_deadline_seconds=14400,
    ):
        responses = []
        for name, command in name_and_commands_map.items():
            print(command)
            response = self.launch_job(
                name=name,
                jobgroup=jobgroup,
                container_image=container_image,
                command=command,
                cpus=cpus,
                memory=memory,
                gpus=gpus,
                namespace=namespace,
                env_vars=env_vars,
                container_name=container_name,
                service_account_name=service_account_name,
                gke_node_pool_name=gke_node_pool_name,
                restart_policy=restart_policy,
                attempts=attempts,
                host_ipc=host_ipc,
                active_deadline_seconds=active_deadline_seconds,
            )
            responses.append(response)
        return responses

    def get_jobgroup_status(self, namespace, jobgroup):
        if not self.credentials.valid:
            self.api_instance = self.get_instance()
        response = self.api_instance.list_namespaced_job(
            namespace=namespace, label_selector=f"jobgroup={jobgroup}"
        )
        active_job_names = []
        failed_job_names = []
        succeeded_job_names = []
        active_commands = []
        failed_commands = []
        succeeded_commands = []
        for item in response.items:
            name = item.metadata.name
            if item.status.succeeded:
                succeeded_job_names.append(name)
                succeeded_commands.append(
                    item.spec.template.spec.containers[0].args
                )
            elif item.status.failed:
                failed_job_names.append(name)
                failed_commands.append(
                    item.spec.template.spec.containers[0].args
                )
            elif item.status.active:
                active_job_names.append(name)
                active_commands.append(
                    item.spec.template.spec.containers[0].args
                )
        error_logs = []
        if len(active_job_names) == 0 and len(failed_job_names) != 0:
            error_logs = self._get_error_logs(response.items, namespace)
        return {
            "active": len(active_job_names),
            "failed": len(failed_job_names),
            "succeeded": len(succeeded_job_names),
            "total": len(active_job_names)
            + len(failed_job_names)
            + len(succeeded_job_names),
            "active_commands": active_commands,
            "failed_commands": failed_commands,
            "succeeded_commands": succeeded_commands,
            "error_logs": error_logs,
        }

    def delete_jobs(self, namespace, jobgroup):
        _continue = None
        while True:
            response = self.api_instance.delete_collection_namespaced_job(
                namespace=namespace,
                label_selector=f"jobgroup={jobgroup}",
                propagation_policy="Background",
                _continue=_continue,
            )
            if response.metadata._continue is None:
                return response
            else:
                _continue = response.metadata._continue

    def _get_error_logs(self, items, namespace):
        error_logs = []
        for item in items:
            if item.status.failed:
                core_v1 = client.CoreV1Api()
                job_def = self.api_instance.read_namespaced_job(
                    name=item.metadata.name, namespace=namespace
                )
                controller_uid = job_def.metadata.labels["controller-uid"]
                pod_label_selector = "controller-uid=" + controller_uid
                pods_list = core_v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=pod_label_selector,
                    timeout_seconds=10,
                )
                for pod_item in pods_list.items:
                    pod_name = pod_item.metadata.name
                    error_logs.append(
                        f"Job: {item.metadata.name} Pod name: {pod_name}"
                    )
                    try:
                        error_logs.append(
                            core_v1.read_namespaced_pod_log(
                                name=pod_name,
                                namespace=namespace,
                                pretty=True,
                                tail_lines=5,
                            )
                        )
                    except ApiException as e:
                        error_logs.append(
                            f"Exception when calling CoreV1Api->read_namespaced_pod_log: {e}"
                        )

        return error_logs
