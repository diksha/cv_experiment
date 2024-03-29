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

from experimental.nasha.spill_segmentation.transforms.albumentation import (
    AlbumentationsTransforms,
)

REGISTRY = {
    "albumentations": AlbumentationsTransforms(),
}


def get_transform(name):
    """Gets the specified model backbone

    Args:
        name (str): name of model type for the model backbone

    Returns:
        function: model for training
    """
    return REGISTRY[name]
