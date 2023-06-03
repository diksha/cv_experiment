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

# Please remove as soon as we have some of this spaghetti removed :)
# Similar lines
#

import torch

# All this should be replaced with the new registry defined in `core/ml/data/generation/common/`
# until then these will be ignored:
# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(pylint/W9012)
# trunk-ignore-all(pylint/C0116)
import torchvision
from torchvision.transforms import (
    AutoAugment,
    CenterCrop,
    ColorJitter,
    GaussianBlur,
    Normalize,
    RandomAdjustSharpness,
    RandomAutocontrast,
    RandomErasing,
    RandomHorizontalFlip,
    RandomPosterize,
    RandomResizedCrop,
    Resize,
    ToPILImage,
    ToTensor,
)
from transformers import ViTFeatureExtractor


# TODO: see where we want to put this to productionize these transforms
class ViTFeatureExtractorTransform:
    """
    Feature extractor for ViT based on the pretrained hugging face model
    """

    def __init__(
        self, pretrained_model: str = "google/vit-base-patch16-224-in21k"
    ):
        """
        Initializes the vit feature extractor transform

        Args:
            pretrained_model (str, optional): The pretrained model to be loaded in from hugging face
                                        Defaults to "google/vit-base-patch16-224-in21k".
        """
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            pretrained_model
        )

    def __call__(self, batch: torch.tensor) -> torch.tensor:
        """
        Runs the transform across the batch and returns the pixel values

        Args:
            batch (torch.tensor): the current input batch

        Returns:
            torch.tensor: the postprocessed batch
        """
        return self.feature_extractor(batch, return_tensors="pt")[
            "pixel_values"
        ][0]


REGISTRY = {
    Resize.__name__: Resize,
    RandomResizedCrop.__name__: RandomResizedCrop,
    RandomHorizontalFlip.__name__: RandomHorizontalFlip,
    ToTensor.__name__: ToTensor,
    Normalize.__name__: Normalize,
    CenterCrop.__name__: CenterCrop,
    AutoAugment.__name__: AutoAugment,
    ViTFeatureExtractorTransform.__name__: ViTFeatureExtractorTransform,
    ToPILImage.__name__: ToPILImage,
    RandomAutocontrast.__name__: RandomAutocontrast,
    RandomAdjustSharpness.__name__: RandomAdjustSharpness,
    RandomPosterize.__name__: RandomPosterize,
    GaussianBlur.__name__: GaussianBlur,
    ColorJitter.__name__: ColorJitter,
    RandomErasing.__name__: RandomErasing,
}


def get_transforms(transforms_json_list):
    ret_transforms = []
    for transform in transforms_json_list:

        ret_transforms.append(
            REGISTRY[transform["name"]](**transform["params"])
        )
    return torchvision.transforms.Compose(ret_transforms)
