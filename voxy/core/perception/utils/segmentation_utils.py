from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from core.structs.frame import FrameSegmentCategory

IMG_DIM_LANDSCPAE = (1088, 608)
IMG_DIM_PORTRAIT = (608, 1088)


def preprocess_segmentation_input(img: np.ndarray) -> torch.Tensor:
    """Process image to and create torch tensor from it

    Args:
        img (np.ndarray): input image

    Returns:
        torch.Tensor: output tensor
    """
    img_transform = transforms.ToTensor()
    img_size = img.shape
    if img_size[0] > img_size[1]:
        img_size = IMG_DIM_LANDSCPAE
    else:
        img_size = IMG_DIM_PORTRAIT
    img = cv2.resize(img, (img_size[1], img_size[0]), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32")
    img_tensor = img_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    return img_tensor


def post_process_segment(
    img: np.ndarray,
    target_class: FrameSegmentCategory,
    min_pixel_size: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    """Post process to remove small connected components and get actor bboxes

    Args:
        img (np.ndarray): input image
        target_class (FrameSegmentCategory): class of interest
        min_pixel_size (int): minimum pixel size

    Returns:
        Tuple[np.ndarray, torch.Tensor]: filtered image without instances < min_pixel_size,
        actor bboxes
    """

    def convert_stat_to_bounding_box_temp(stat: np.ndarray) -> torch.Tensor:
        """
        Converts a connected components statistic into a pytorch
        tensor detection in yolo format

        Args:
            stat (np.ndarray): the connected component statistic

        Returns:
            torch.Tensor: the bounding box for the connected component
        """

        return torch.tensor(
            [
                stat[cv2.CC_STAT_LEFT],
                stat[cv2.CC_STAT_TOP],
                stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH],
                stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT],
                1.0,
                1.0,
            ],
            requires_grad=False,
        )

    img_copy = np.copy(img)
    img_copy[img_copy != target_class.value] = 0
    img_copy[img_copy == target_class.value] = 1
    img_copy = img_copy.astype("uint8")
    # Finding Connected components
    (
        _,
        im_with_separated_blobs,
        stats,
        _,
    ) = cv2.connectedComponentsWithStats(img_copy)
    # remove first connected components
    # the first connected componets is for background part
    stats = stats[1:]
    # if there is no connected components we
    # output the input segmentation mask and
    # an empty tensor for bounding box
    if len(stats) == 0:
        return img, torch.zeros(0, 6)
    actors_bbox = []
    for blob, stat in enumerate(stats):
        # if size of the connected components satisfy the min pixel size
        if stat[cv2.CC_STAT_AREA] >= min_pixel_size:
            img[im_with_separated_blobs == blob + 1] = target_class.value
            actors_bbox.append(convert_stat_to_bounding_box_temp(stat))
        else:
            img[
                im_with_separated_blobs == blob + 1
            ] = FrameSegmentCategory.UNKNOWN.value
    if len(actors_bbox) > 0:
        actors_bbox = torch.stack(actors_bbox)
    else:
        actors_bbox = torch.zeros(0, 6)
    return img, actors_bbox


def update_segment_class_ids(
    frame_segment: np.ndarray, frame_segments_category_to_class_id: dict
) -> np.ndarray:
    """Update class id based on voxel ids

    Args:
        frame_segment (np.ndarray): input image
        frame_segments_category_to_class_id (dict): categories to class id dict
    Returns:
        np.ndarray: output segment
    """
    new_frame_segment = np.copy(frame_segment)
    for (
        category,
        class_id,
    ) in frame_segments_category_to_class_id.items():
        new_frame_segment[frame_segment == class_id] = category.value
    return new_frame_segment
