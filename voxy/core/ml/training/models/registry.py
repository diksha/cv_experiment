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

from core.ml.training.models.classifiermodels import (
    AttentionResnet50,
    ModelEMA,
    VanillaResnet50Deprecated,
)
from core.ml.training.models.vanilla_resnet import VanillaResnet50
from core.ml.training.models.yolo_model import YoloModel

# trunk-ignore-all(pylint/C0116)
# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(pylint/W9012)

# TODO(diksha): Move these to REGISTRY
REGISTRY_DEPRECATED = {
    "vanilla_resnet_deprecated": VanillaResnet50Deprecated,
    "attention_resnet": AttentionResnet50,
    "ema_model": ModelEMA,
}


REGISTRY = {"YoloModel": YoloModel, "VanillaResnet": VanillaResnet50}


def get_model_deprecated(name):
    return REGISTRY_DEPRECATED[name]
