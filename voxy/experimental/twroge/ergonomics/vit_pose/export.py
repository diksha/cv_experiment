import argparse
import os
import typing
from typing import Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision.transforms.functional import (
    InterpolationMode,
    resize,
    to_tensor,
)

from third_party.vit_pose.configs.ViTPose_base_coco_256x192 import data_cfg
from third_party.vit_pose.configs.ViTPose_base_coco_256x192 import (
    model as model_cfg,
)
from third_party.vit_pose.models.model import ViTPose
from third_party.vit_pose.utils.top_down_eval import keypoints_from_heatmaps
from third_party.vit_pose.utils.visualization import (
    draw_points_and_skeleton,
    joints_dict,
)


# TODO: move to a utils file
def preprocess_image(
    image: torch.Tensor, image_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Preprocesses the image

    Args:
        image (torch.Tensor): the original image tensor
        image_size (Tuple[int, int]): the image size

    Returns:
        torch.Tensor: the resized image
    """
    resized_image = (
        resize(
            img=image,
            size=[int(image_size[1]), int(image_size[0])],
            interpolation=InterpolationMode.BILINEAR,
        )
        .unsqueeze(0)
        .to("cuda")
    )
    return resized_image


def post_process(
    heatmap: torch.tensor, original_size: typing.Tuple[int, int]
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Post processes the prediction

    Args:
        heatmap (torch.tensor): the heatmap from vit pose
        original_size (typing.Tuple[int, int]): the original size (width, height)

    Returns:
        torch.Tensor: the keypoints and probabilities
    """
    org_w, org_h = original_size
    center = np.array([[org_w // 2, org_h // 2]])
    scale = np.array([[org_w, org_h]])
    keypoints, probabilities = keypoints_from_heatmaps(
        heatmap, center=center, scale=scale, unbiased=True, use_udp=True
    )
    return keypoints, probabilities


class ViTPoseModel:
    def __init__(
        self, model_path: str, device: str = "cuda", jit: bool = False
    ):
        if not jit:
            self.model = ViTPose(model_cfg)
            self.model.eval().to(device)
            checkpoint = torch.load(  # trunk-ignore(semgrep/trailofbits.python.pickles-in-pytorch.pickles-in-pytorch,pylint/C0301)
                model_path
            )
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model = torch.jit.load(model_path)
        self.image_size = data_cfg["image_size"]

    def __call__(self, image: torch.Tensor):
        with torch.no_grad():
            preprocessed = preprocess_image(image, self.image_size)
            heatmaps = self.model(preprocessed).detach().cpu().numpy()
            return post_process(heatmaps, image.shape[1:3][::-1])


def get_args() -> argparse.Namespace:
    """
    Gets the arguments from the command line.

    Returns:
        argparse.Namespace: the commandline args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def draw(
    img: Image,
    points: torch.Tensor,
    prob: torch.Tensor,
    output_dir: str,
    output_name: str = "vit_pose_result.jpg",
):
    """
    Draws the keypoints on the image

    Args:
        img: the image to draw on
        points: the points
        prob: the probability
        output_dir: the output directory
        output_name: the output filename

    Raises:
        RuntimeError: if the image could not be saved
    """
    points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    for pid, point in enumerate(points):
        img = np.array(img)[:, :, ::-1]  # RGB to BGR for cv2 modules
        drawn_image = img.copy()
        drawn_image = draw_points_and_skeleton(
            drawn_image,
            point,
            joints_dict()["coco"]["skeleton"],
            person_index=pid,
            points_color_palette="gist_rainbow",
            skeleton_color_palette="jet",
            points_palette_samples=10,
            confidence_threshold=0.4,
        )
        save_name = os.path.join(output_dir, output_name)
        result = cv2.imwrite(save_name, drawn_image)
        if not result:
            raise RuntimeError("Image could not be saved")
        logger.success(f"Saved image to {save_name}")


def main(args: argparse.Namespace):
    vit_pose = ViTPoseModel(args.model_path, args.device)
    logger.success("Loaded model")

    img = Image.open(args.image_path)
    points, probs = vit_pose(to_tensor(img))
    draw(img, points, probs, args.output_path)

    # try to trace the model
    preprocessed = preprocess_image(to_tensor(img), vit_pose.image_size)
    logger.info("Tracing model")
    traced_model = torch.jit.trace(vit_pose.model, preprocessed)
    output_path = os.path.join(args.output_path, "vit_pose_model-jit.pt")
    torch.jit.save(traced_model, output_path)
    logger.info(f"Saved traced model to : {output_path}")

    # try to run the jit model now
    jit_vit_pose = ViTPoseModel(output_path, args.device, jit=True)
    points, probs = jit_vit_pose(to_tensor(img))
    draw(
        img,
        points,
        probs,
        args.output_path,
        output_name="jit_vit_pose_result.jpg",
    )


if __name__ == "__main__":
    main(get_args())
