#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import torchvision
from torch import nn


class VanillaResnet50Module(nn.Module):
    """
    Vanilla Resnet 50 Module. This is a simple
    wrapper around the pytorch resnet50 model
    """

    def __init__(self, num_classes: int, freeze_layers: bool):
        """
        Initializes the module

        Args:
            num_classes (int): the number of classes to train
            freeze_layers (bool): whether or not to freeze
                                  layers for gradients
        """
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = not freeze_layers
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
