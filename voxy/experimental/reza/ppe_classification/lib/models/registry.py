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

from experimental.reza.ppe_classification.lib.models.ppemodels import VanillaResnet50, AttentionResnet50

REGISTRY = {
    "vanilla_resnet": VanillaResnet50,
    "attention_resnet": AttentionResnet50
}

def get_model(name):
    return REGISTRY[name]
