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
from experimental.nasha.spill_segmentation.optimizer.ranger import (
    RangerOptimizer,
)

RangerOptimizer = RangerOptimizer()
REGISTRY = {"ranger_optimzer": RangerOptimizer.ranger_optimizer()}


def get_optimizer(name):
    """Fetches the optmizer specified by config

    Args:
        name (str): name of optmization function

    Returns:
        function: an optmization function
    """
    return REGISTRY[name]
