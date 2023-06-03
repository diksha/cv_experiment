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
import torch.nn.functional as F
from torchvision.models import mobilenet as vmn
from torchvision.models import resnet as vrn

from .mobilenet import MobileNet
from .resnet import ResNet


class FPN(nn.Module):
    "Feature Pyramid Network - https://arxiv.org/abs/1612.03144"

    def __init__(self, features):
        super().__init__()

        self.stride = 128
        self.features = features

        if isinstance(features, ResNet):
            is_light = features.bottleneck == vrn.BasicBlock
            channels = [128, 256, 512] if is_light else [512, 1024, 2048]
        elif isinstance(features, MobileNet):
            channels = [32, 96, 320]

        self.lateral3 = nn.Conv2d(channels[0], 256, 1)
        self.lateral4 = nn.Conv2d(channels[1], 256, 1)
        self.lateral5 = nn.Conv2d(channels[2], 256, 1)
        self.pyramid6 = nn.Conv2d(channels[2], 256, 3, stride=2, padding=1)
        self.pyramid7 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, 3, padding=1)

    def initialize(self):
        def init_layer(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        self.apply(init_layer)

        self.features.initialize()

    def forward(self, x):
        c3, c4, c5 = self.features(x)

        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4)
        p4 = F.interpolate(p5, scale_factor=2) + p4
        p3 = self.lateral3(c3)
        p3 = F.interpolate(p4, scale_factor=2) + p3

        p6 = self.pyramid6(c5)
        p7 = self.pyramid7(F.relu(p6))

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return [p3, p4, p5, p6, p7]


def ResNet18FPN():
    return FPN(
        ResNet(
            layers=[2, 2, 2, 2],
            bottleneck=vrn.BasicBlock,
            outputs=[3, 4, 5],
            url=vrn.model_urls["resnet18"],
        )
    )


def ResNet34FPN():
    return FPN(
        ResNet(
            layers=[3, 4, 6, 3],
            bottleneck=vrn.BasicBlock,
            outputs=[3, 4, 5],
            url=vrn.model_urls["resnet34"],
        )
    )


def ResNet50FPN():
    return FPN(
        ResNet(
            layers=[3, 4, 6, 3],
            bottleneck=vrn.Bottleneck,
            outputs=[3, 4, 5],
            url=vrn.model_urls["resnet50"],
        )
    )


def ResNet101FPN():
    return FPN(
        ResNet(
            layers=[3, 4, 23, 3],
            bottleneck=vrn.Bottleneck,
            outputs=[3, 4, 5],
            url=vrn.model_urls["resnet101"],
        )
    )


def ResNet152FPN():
    return FPN(
        ResNet(
            layers=[3, 8, 36, 3],
            bottleneck=vrn.Bottleneck,
            outputs=[3, 4, 5],
            url=vrn.model_urls["resnet152"],
        )
    )


def ResNeXt50_32x4dFPN():
    return FPN(
        ResNet(
            layers=[3, 4, 6, 3],
            bottleneck=vrn.Bottleneck,
            outputs=[3, 4, 5],
            groups=32,
            width_per_group=4,
            url=vrn.model_urls["resnext50_32x4d"],
        )
    )


def ResNeXt101_32x8dFPN():
    return FPN(
        ResNet(
            layers=[3, 4, 23, 3],
            bottleneck=vrn.Bottleneck,
            outputs=[3, 4, 5],
            groups=32,
            width_per_group=8,
            url=vrn.model_urls["resnext101_32x8d"],
        )
    )


def MobileNetV2FPN():
    return FPN(MobileNet(outputs=[6, 13, 17], url=vmn.model_urls["mobilenet_v2"]))
