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
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PerceptionPipelineConfig:
    """Configurations for the perception portion of the pipeline

    Attributes
    ----------
    inference_cluster_size:
        Inferences will be executed on a Ray cluster. This parameter specifies
        how many workers (including the head) will be in that cluster.
    video_uuid_filter:
        If this is not None OR empty, only videos with the provided UUIDs will have
        perception executed on them
    """

    inference_cluster_size: int
    video_uuid_filter: Optional[List[str]] = None
