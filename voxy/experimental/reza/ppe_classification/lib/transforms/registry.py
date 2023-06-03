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
import torchvision
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, CenterCrop, Resize

REGISTRY = {
    "Resize": Resize,
    "RandomResizedCrop": RandomResizedCrop,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "ToTensor": ToTensor,
    "Normalize": Normalize,
    "CenterCrop": CenterCrop,
}


def get_transforms(transforms_json_list):
    ret_transforms = []
    for transform in transforms_json_list:
       
        ret_transforms.append(
            REGISTRY[transform["name"]](**transform["params"])
        )
    return torchvision.transforms.Compose(ret_transforms)
