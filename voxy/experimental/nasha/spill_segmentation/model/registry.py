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

from experimental.nasha.spill_segmentation.model.resnet import Resnet
from experimental.nasha.spill_segmentation.model.smp import SmpModel
from experimental.nasha.spill_segmentation.model.squeezenet import Squeezenet

Resnet = Resnet()
Squeezenet = Squeezenet()
REGISTRY = {
    "resnet18": Resnet.resnet18(),
    "resnet34": Resnet.resnet34(),
    "resnet50": Resnet.resnet50(),
    "resnet101": Resnet.resnet101(),
    "resnet152": Resnet.resnet152(),
    "squeezenet1_0": Squeezenet.squeezenet1_0(),
    "squeezenet1_1": Squeezenet.squeezenet1_1(),
    "smp": SmpModel,
}


def get_model(name):
    """Gets the specified model backbone

    Args:
        name (str): name of model type for the model backbone

    Returns:
        function: model for training
    """
    return REGISTRY[name]
