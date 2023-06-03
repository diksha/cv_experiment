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

# Resource guidelines
# 1. Cpu:Memory multiplier should be around 1:4, based on the nodes we currently use on the cluster.
# 2. Specify cpu memory as ~900m instead of 1 cpu and memory as ~3600M instead of 4Gi.
# 3. For GPU workloads specify gpu instance with node selector.
# 4. Use maximum of 90% of the memory on the node, to leave some room for other startup jobs.
"""Common compute resource configurations used across multiple Sematic pipelines"""
from sematic import (
    KubernetesResourceRequirements,
    KubernetesToleration,
    KubernetesTolerationEffect,
    ResourceRequirements,
)
from sematic.ee.ray import RayNodeConfig


def resources(
    cpu: str,
    memory: str,
    gpu_instance: str = None,
    mount_expanded_shared_memory=False,
) -> ResourceRequirements:
    """Create sematic resource requests given specifications on what resources are needed

    Args:
        cpu: The CPU specification.

        memory: The memory required.

        gpu_instance: Instance type of gpu. Example: g4dn.2xlarge, g4dn.xlarge
        More information here: https://aws.amazon.com/ec2/instance-types/g4/

        mount_expanded_shared_memory(bool): To expand shared memory to 50% of memory of pod

    Returns:
        The equivalent Sematic resource requirements, tailored for Voxel's Kubernetes
        deployment, with GCP credentials attached.

    Raises:
        ValueError: If the number of GPUs is something other than 0 or 1
    """
    if not gpu_instance:
        return ResourceRequirements(
            kubernetes=KubernetesResourceRequirements(
                requests={"cpu": cpu, "memory": memory},
                mount_expanded_shared_memory=mount_expanded_shared_memory,
            )
        )
    return ResourceRequirements(
        kubernetes=KubernetesResourceRequirements(
            node_selector={"node.kubernetes.io/instance-type": gpu_instance},
            tolerations=[
                KubernetesToleration(
                    key="nvidia.com/gpu",
                    value="true",
                    effect=KubernetesTolerationEffect.NoSchedule,
                )
            ],
            requests={"cpu": cpu, "memory": memory},
            mount_expanded_shared_memory=mount_expanded_shared_memory,
        )
    )


# Default resources for sematic pipelines
# m -> cpu in milli
# M -> memory in MBs
# 1x -> how many similar pods simultaneously can run on the node basically shared
# Default resources for sematic pipelines


# 2x large instance: 8 cpu / 32 gb memory
# 1x large instance: 4 cpu / 16 gb memory
GPU_1CPU_4GB_8x = resources(
    cpu="900m",
    memory="3600M",
    gpu_instance="g5.2xlarge",
    mount_expanded_shared_memory=True,
)
GPU_1CPU_4GB_4x = resources(
    cpu="900m",
    memory="3600M",
    gpu_instance="g5.xlarge",
    mount_expanded_shared_memory=True,
)
GPU_4CPU_16GB_2x = resources(
    cpu="3600m",
    memory="7200M",
    gpu_instance="g5.2xlarge",
    mount_expanded_shared_memory=True,
)
GPU_4CPU_16GB_1x = resources(
    cpu="3600m",
    memory="14400M",
    gpu_instance="g5.xlarge",
    mount_expanded_shared_memory=True,
)
GPU_8CPU_32GB_1x = resources(
    cpu="7200m",
    memory="28800M",
    gpu_instance="g5.2xlarge",
    mount_expanded_shared_memory=True,
)
GPU_16CPU_64GB_1x = resources(
    cpu="14400m",
    memory="57600M",
    gpu_instance="g5.4xlarge",
    mount_expanded_shared_memory=True,
)
CPU_1CORE_4GB = resources(cpu="900m", memory="3600M")
CPU_2CORE_8GB = resources(cpu="1800m", memory="7200M")
CPU_4CORE_16GB = resources(cpu="3600m", memory="14400M")
CPU_8CORE_32GB = resources(cpu="7200m", memory="28800M")
# Sematic uses a combination of Gi and M for memory, so 14.8 gb = (1024*14.8)M
RAY_NODE_GPU_4CPU_16GB = RayNodeConfig(cpu=3.6, memory_gb=14.4, gpu_count=1)
RAY_NODE_2CPU_8GB = RayNodeConfig(cpu=1.8, memory_gb=7.2)
