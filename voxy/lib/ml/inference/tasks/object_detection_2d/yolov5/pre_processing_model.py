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

from typing import Tuple

import torch
from torchvision.transforms.functional import InterpolationMode, pad, resize


@torch.jit.script
def letterbox(
    img: torch.Tensor,
    new_shape: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scripted letterbox function derived from
    https://github.com/voxel-ai/yolov5/blob/master/utils/augmentations.py
    Args:
        img (torch.Tensor): input image (NxCxHxW)
        new_shape (torch.Tensor): shape to resize (NxHxW)
    Returns:
        typing.Any: tuple with letterboxed image, scale, and offset
    """
    shape = torch.tensor(img.shape[2:], device=img.device)
    # Scale ratio (new / old)
    scale_ratio = (new_shape / shape).min()
    new_unpad = (shape * scale_ratio).round().int()
    if not torch.equal(shape, new_unpad):  # resize
        img = resize(
            img=img,
            size=[int(new_unpad[0]), int(new_unpad[1])],
            interpolation=InterpolationMode.BILINEAR,
        )
    delta_height_width = (new_shape - new_unpad) / 2
    top_left = (delta_height_width - 0.1).round().int()
    bottom_right = (delta_height_width + 0.1).round().int()
    img = pad(
        img=img,
        padding=[
            int(top_left[0, 1]),
            int(top_left[0, 0]),
            int(bottom_right[0, 1]),
            int(bottom_right[0, 0]),
        ],
        fill=114,
        padding_mode="constant",
    )
    return (
        img,
        scale_ratio.expand(delta_height_width.shape),
        delta_height_width.flip(1),
    )


@torch.jit.script
def preprocess_image(
    image_batch: torch.Tensor,
    input_shape: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """preprocess_image.

    preprocesses the image so it is ready for inference

    Args:
        image_batch (torch.Tensor): raw input image (NHWC)
        input_shape (torch.Tensor): input shape to transorm image to (HxW)
    Returns:
        torch.Tuple: tuple containing the preprocessed tensor, along with the scale and
            offset tensors of the transformation with respect to the original
    """
    image_batch_nchw = image_batch.permute(0, 3, 1, 2)
    letterboxed_batch, scale, offset = letterbox(image_batch_nchw, input_shape)
    rgb_nchw = letterboxed_batch.flip(1)
    preprocessed_batch = rgb_nchw.half() / 255.0
    return preprocessed_batch, offset, scale
