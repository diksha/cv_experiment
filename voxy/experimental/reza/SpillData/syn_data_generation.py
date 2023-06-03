import argparse
import json
import os
import shutil
import tempfile
from typing import Tuple

import cv2
import numpy as np
from loguru import logger

from core.utils.aws_utils import (
    download_to_file,
    get_bucket_path_from_s3_uri,
    glob_from_bucket,
    upload_cv2_imageobj_to_s3,
)

SPILL_SAVE_RELATIVE_PATH = (
    "common/spill_data"  # The Path to strore Spill Data As the Default
)


def rgb2mask(img, color2index):
    """Rgb Image to label image converter

    Args:
        img (_type_): input rgb image
        color2index (_type_): color to label map

    Returns:
        _type_: indexed label image
    """
    img_id = img.dot(np.power(256, [[0], [1], [2]])).squeeze(-1)
    color_values = np.unique(img_id)
    mask = np.zeros(img_id.shape)
    for color in color_values:
        if tuple(img[img_id == color][0]) in color2index.keys():
            mask[img_id == color] = color2index[tuple(img[img_id == color][0])]
    return mask


def get_data_from_s3(dataset_path):
    """Get data from a s3 path containing multiple jobs

    Args:
        dataset_path (_type_): path to a s3 bucket containing multiple jobs
    """
    bucket, s3_relative_path = get_bucket_path_from_s3_uri(dataset_path)
    list_files = glob_from_bucket(
        bucket,
        prefix=s3_relative_path,
        extensions=("zip"),
    )
    for job_zip in list_files:
        with tempfile.NamedTemporaryFile(suffix=".zip") as temp_file:
            job_name = job_zip.split("/")[-1][:-4]
            download_to_file(
                bucket,
                job_zip,
                f"{temp_file.name}",
            )
            with tempfile.TemporaryDirectory() as zip_extract_dir:
                shutil.unpack_archive(f"{temp_file.name}", zip_extract_dir)
                if os.path.exists(f"{zip_extract_dir}/{job_name}/job.json"):
                    generate_dataset(f"{zip_extract_dir}/{job_name}")


def get_color_index(categories: list) -> dict:
    """Get color to label index for categories of objects

    Args:
        categories (list): list of object categories

    Returns:
        dict: color to label map
    """
    color_2_idx = {(0, 0, 0): 0}
    for cat in categories:
        rgb_color = tuple(int(item * 255) for item in cat["color"])
        if cat["name"] == "spill":
            color_2_idx[rgb_color] = 1
        else:
            color_2_idx[rgb_color] = 0
    return color_2_idx


def get_camera_uuid(scene: str) -> str:
    """Get camera uuid from the scene name

    Args:
        scene (str): scene na,e

    Returns:
        str: camera_uuid
    """
    if "roomf1" in scene:
        scene = "uscold-laredo-room_f1"
    camera_path = "/".join(scene.split("-"))
    return camera_path


def save_images_to_s3(
    camera_uuid: str,
    folder_name: str,
    frame_id: int,
    image: np.ndarray,
    mask: np.ndarray,
):
    """Save image and indexed image in s3

    Args:
        camera_uuid (str): camera uuid
        folder_name (str): dataset name for a scnene
        frame_id (int): frame id
        image (np.ndarray): image
        mask (np.ndarray): indexed label
    """
    img_name = os.path.join(
        "s3://voxel-users",
        SPILL_SAVE_RELATIVE_PATH,
        camera_uuid,
        "img/synthetic/positive",
        f"{folder_name}_img_00000{frame_id}.png",
    )
    anno_name = os.path.join(
        "s3://voxel-users",
        SPILL_SAVE_RELATIVE_PATH,
        camera_uuid,
        "annotation/synthetic/positive",
        f"{folder_name}_img_00000{frame_id}.png",
    )
    upload_cv2_imageobj_to_s3(img_name, image)
    upload_cv2_imageobj_to_s3(anno_name, mask)


def get_image_size(image: np.ndarray) -> Tuple[int, int]:
    """Update image size
    Args:
        image (np.ndarray): image

    Returns:
        Tuple[int, int]: tuple of width and height
    """
    if image.shape[0] > image.shape[1]:
        width, height = 610, 1080
    else:
        width, height = 1080, 610
    return width, height


def get_index_image(
    img_path: str, img_idx: int, color_2_label: dict
) -> np.ndarray:
    """Get indexed image

    Args:
        img_path (str): image path
        img_idx (int): image number
        color_2_label (dict): color to label dict

    Returns:
        np.ndarray: indexed image
    """
    label_image = cv2.imread(f"{img_path}/image.00000{img_idx}.cseg.png")
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
    index_image = rgb2mask(label_image, color_2_label)
    return index_image


def generate_dataset(path: str):
    """Generate the dataset

    Args:
        path (str): dataset subname for each scene
    """
    with open(f"{path}/video.rgb.json", "r", encoding="UTF-8") as label:
        data = json.load(label)
    color_2_label = get_color_index(data["categories"])

    shutil.unpack_archive(f"{path}/video.rgb.zip", path)
    with open(f"{path}/job.json", "r", encoding="UTF-8") as job_f:
        job = json.load(job_f)
    camera_uuid = get_camera_uuid(job["params"]["scene"])
    folder_name = os.path.split(path)[-1]
    logger.info(f"Processing Data of {folder_name}")
    imagelists = [
        file
        for file in os.listdir(path)
        if file.endswith(".png")
        and "seg" not in file
        and "video_preview" not in file
    ]
    if len(imagelists) > 0:
        for img_name in imagelists:
            image = cv2.imread(f"{path}/{img_name}")
            count = int(img_name.split(".")[1])
            width, height = get_image_size(image)
            index_image = get_index_image(path, count, color_2_label)
            image = cv2.resize(
                image, (width, height), interpolation=cv2.INTER_NEAREST
            )
            index_image = cv2.resize(
                index_image, (width, height), interpolation=cv2.INTER_NEAREST
            ).astype("uint8")
            save_images_to_s3(
                camera_uuid, folder_name, count, image, index_image
            )
    else:

        vidcap = cv2.VideoCapture(f"{path}/video.rgb.mp4")
        success, image = vidcap.read()
        width, height = get_image_size(image)
        count = 0
        while success:
            index_image = get_index_image(path, count, color_2_label)
            image = cv2.resize(
                image, (width, height), interpolation=cv2.INTER_NEAREST
            )
            index_image = cv2.resize(
                index_image, (width, height), interpolation=cv2.INTER_NEAREST
            ).astype("uint8")
            save_images_to_s3(
                camera_uuid, folder_name, count, image, index_image
            )
            success, image = vidcap.read()
            count += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_s3_path", "--s3_directory", type=str)
    args = parser.parse_args()
    if args.data_s3_path:
        get_data_from_s3(args.data_s3_path)
