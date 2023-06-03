#
# Copyright 2020-2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import typing

import torch
from transformers import ViTFeatureExtractor


def crop_actors(
    actors_xyxy: typing.List[torch.Tensor],
    frame: torch.Tensor,
    padding: int,
) -> typing.List[torch.Tensor]:
    """Crops frame tensor with actor bounding boxes
    Args:
        actors_xyxy (typing.List[torch.Tensor]): list of actor tensors
            denoting their bbox location (tl_x, tl_y, bl_x, bl_y)
        frame (np.ndarray): frame of image being processed (HxWxC)
        padding (int): constant padding to apply around actor
    Returns:
        typing.List[torch.Tensor]: cropped tensors
    """
    cropped_images = []
    for xyxy in actors_xyxy:
        cropped_image = frame[
            max(0, int(xyxy[1] - padding)) : min(
                int(frame.shape[0]), int(xyxy[3] + padding)
            ),
            max(0, int(xyxy[0] - padding)) : min(
                int(frame.shape[1]), int(xyxy[2] + padding)
            ),
            :,
        ]
        cropped_image_rgb = cropped_image.flip(2)
        cropped_images.append(cropped_image_rgb)
    return cropped_images


def preprocess_inputs(
    actors_xyxy: typing.List[torch.Tensor],
    frame: torch.Tensor,
    feature_extractor: "ViTFeatureExtractor",
    padding: int,
    device: torch.device,
) -> torch.Tensor:
    """Preprocess inputs for PPE ViT based image classification tasks
    Args:
        actors_xyxy (typing.List[torch.Tensor]): list of actor tensors
            denoting their bbox location (tl_x, tl_y, bl_x, bl_y)
        frame (np.ndarray): frame of image being processed (HxWxC)
        feature_extractor (ViTFeatureExtractor): ViTFeatureExtractor
        padding (int): constant padding to apply around actor
        device (torch.device): device to put tensor in memory
    Returns:
        torch.Tensor: ViT feature tensor
    """
    cropped_images = crop_actors(actors_xyxy, frame, padding)
    return feature_extractor(cropped_images, return_tensors="pt")[
        "pixel_values"
    ].to(device)
