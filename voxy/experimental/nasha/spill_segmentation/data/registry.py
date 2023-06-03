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

from experimental.nasha.spill_segmentation.data.smp_dataset import (
    SegmentationDataset,
)

REGISTRY = {
    "smp_segmentation": SegmentationDataset,
}

# trunk-ignore-all(pylint/W9012)
# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(pylint/C0116)


def get_dataset(name):
    return REGISTRY[name]
