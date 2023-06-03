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
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models import mobilenet as vmn


class MobileNet(vmn.MobileNetV2):
    "MobileNetV2: Inverted Residuals and Linear Bottlenecks - https://arxiv.org/abs/1801.04381"

    def __init__(self, outputs=[18], url=None):
        self.stride = 128
        self.url = url
        super().__init__()
        self.outputs = outputs

    def initialize(self):
        if self.url:
            self.load_state_dict(model_zoo.load_url(self.url))

    def forward(self, x):
        outputs = []
        for indx, feat in enumerate(self.features[:-1]):
            x = feat(x)
            if indx in self.outputs:
                outputs.append(x)
        return outputs
