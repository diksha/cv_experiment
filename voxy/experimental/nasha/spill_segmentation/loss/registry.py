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

from experimental.nasha.spill_segmentation.loss.cross_entropy_loss import (
    CrossEntropyLoss,
)
from experimental.nasha.spill_segmentation.loss.dice_loss import (
    DiceLossGeneral,
)

REGISTRY = {
    "cross_entropy_loss": CrossEntropyLoss(),
    "dice_loss": DiceLossGeneral(reduction="mean"),
}

# trunk-ignore-all(pylint/W9012)
# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(pylint/C0116)


def get_loss(name):
    return REGISTRY[name]
