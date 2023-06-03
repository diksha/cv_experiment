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

from torch import nn
from torchvision import models

# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(pylint/C0116)
# trunk-ignore-all(pylint/W9012)


def resnet50_classification(num_classes, freeze_layers=True, pretrained=True):
    model = models.resnet50(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = not freeze_layers

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
