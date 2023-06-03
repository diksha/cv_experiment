import typing
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode, resize

from third_party.vit_pose.utils.top_down_eval import keypoints_from_heatmaps


def crop_actors(
    actors_xyxy: typing.List[torch.Tensor],
    frame: torch.Tensor,
    padding: int,
) -> typing.Tuple[typing.List[torch.Tensor], torch.Tensor]:
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
    origins_xy = []
    for xyxy in actors_xyxy:
        y_min, y_max = (
            max(0, int(xyxy[1] - padding)),
            min(int(frame.shape[0]), int(xyxy[3] + padding)),
        )

        x_min, x_max = max(0, int(xyxy[0] - padding)), min(
            int(frame.shape[1]), int(xyxy[2] + padding)
        )
        cropped_image = frame[
            y_min:y_max,
            x_min:x_max,
            :,
        ]
        cropped_image_rgb = cropped_image.flip(2)
        origins_xy.append([x_min, y_min])
        cropped_images.append(cropped_image_rgb)
    return cropped_images, np.array(origins_xy)


def preprocess_images(
    images: torch.Tensor, image_size: Tuple[int, int], device: torch.device
) -> torch.Tensor:
    """
    Preprocesses the image

    Args:
        images (torch.Tensor): the original image tensor
        image_size (Tuple[int, int]): the image size
        device (torch.device): the device to put the tensor in memory

    Returns:
        torch.Tensor: the resized image
    """
    resized_images = (
        torch.stack(
            [
                resize(
                    img=image.permute(2, 0, 1),
                    size=[int(image_size[1]), int(image_size[0])],
                    interpolation=InterpolationMode.BILINEAR,
                )
                .to(torch.uint8)
                .to(device)
                for image in images
            ]
        )
        / 255
        - 0.5
    )

    original_sizes = np.array([image.size()[:2][::-1] for image in images])
    return resized_images, original_sizes


def preprocess_frame(
    actors_xyxy: typing.List[torch.Tensor],
    frame: torch.Tensor,
    padding: int,
    target_size: Tuple[int, int],
    device: torch.device,
) -> typing.Tuple[torch.Tensor, np.ndarray]:
    """Preprocess inputs for PPE ViT based image classification tasks
    Args:
        actors_xyxy (typing.List[torch.Tensor]): list of actor tensors
            denoting their bbox location (tl_x, tl_y, bl_x, bl_y)
        frame (np.ndarray): frame of image being processed (HxWxC)
        padding (int): constant padding to apply around actor
        target_size: (Tuple[int, int]): the target size of the image
        device (torch.device): device to put tensor in memory
    Returns:
        typing.Tuple[torch.Tensor, np.ndarray]: the preprocessed image and the
                   original image sizes
    """
    cropped_images, origins = crop_actors(actors_xyxy, frame, padding)
    return (
        preprocess_images(
            cropped_images, image_size=target_size, device=device
        ),
        origins,
    )


def post_process(
    heatmaps: torch.tensor, original_sizes: np.ndarray, origins: np.ndarray
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Post processes the prediction

    Args:
        heatmaps (torch.tensor): the heatmap from vit pose
        original_sizes (np.ndarray): the original size (width, height)
            sizes are listed as [[width, height], [width, height], ...]
        origins (np.ndarray): the origins of the bounding box (top left x, y)

    Returns:
        torch.Tensor: the keypoints and probabilities
    """
    center = original_sizes // 2
    scale = original_sizes
    keypoints, probabilities = keypoints_from_heatmaps(
        heatmaps.cpu().numpy(),
        center=center,
        scale=scale,
        unbiased=True,
        use_udp=True,
    )
    keypoints[:, :, 0] += origins[:, 0, None]
    keypoints[:, :, 1] += origins[:, 1, None]

    return keypoints, probabilities
