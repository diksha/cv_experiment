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
import numpy as np
import torch
import torchvision.transforms.functional
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import ClipValueRange
from batchgenerators.transforms.crop_and_pad_transforms import RandomCropTransform
from batchgenerators.transforms.sample_normalization_transforms import (
    MeanStdNormalizationTransform,
)
from batchgenerators.transforms.spatial_transforms import MirrorTransform


class ComposeTransform(Compose):
    pass


class RandomCropTransform(RandomCropTransform):
    pass


class ClipValueRangeTransform(ClipValueRange):
    pass


class MirrorTransform(MirrorTransform):
    pass


class MeanStdNormalizationTransform(MeanStdNormalizationTransform):
    pass


class ToTensorTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        if data.ndim == 2:  # Add C to HW
            data = data[None, :, :]
        data_dict[self.data_key] = torch.from_numpy(data)

        if seg is not None:
            if seg.ndim == 2:  # Add C to HW
                seg = seg[None, :, :]
            data_dict[self.label_key] = torch.from_numpy(seg)

        return data_dict


class ReduceDimension(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        data_dict[self.data_key] = np.squeeze(data, axis=0)
        if seg is not None:
            data_dict[self.label_key] = np.squeeze(seg, axis=0)

        return data_dict


class ExpandDimension(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        data_dict[self.data_key] = np.expand_dims(data, axis=0)
        if seg is not None:
            data_dict[self.label_key] = np.expand_dims(seg, axis=0)

        return data_dict


# Keep at the bottom of the file such as to have correct overrides.
REGISTRY = {
    "ExpandDimension": ExpandDimension,
    "ReduceDimension": ReduceDimension,
    "ToTensorTransform": ToTensorTransform,
    "MeanStdNormalizationTransform": MeanStdNormalizationTransform,
    "MirrorTransform": MirrorTransform,
    "RandomCropTransform": RandomCropTransform,
    "ClipValueRangeTransform": ClipValueRangeTransform,
}


def get_transforms(transforms_json_list):
    ret_transforms = []
    for trans in transforms_json_list:
        ret_transforms.append(REGISTRY[trans["name"]](**trans["args"]))
    return ret_transforms
