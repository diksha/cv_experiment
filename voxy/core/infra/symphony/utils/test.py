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
from core.infra.symphony.utils.k8s_client import K8sClient

k8s_client = K8sClient()

print(
    k8s_client.get_jobgroup_status(
        namespace="default", jobgroup="yolo-9-3-2021-7"
    )
)
