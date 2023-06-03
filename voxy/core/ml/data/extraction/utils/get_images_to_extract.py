#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import argparse
import json
import tempfile
from typing import List

import sematic
from loguru import logger

from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.utils.aws_utils import download_to_file, glob_from_bucket


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bucket",
        metavar="B",
        type=str,
        default="voxel-datasets",
        help="Bucket where images are stored",
    )
    parser.add_argument(
        "-r",
        "--relative_path",
        metavar="R",
        type=str,
        required=True,
        help="Relative path to images from bucket",
    )
    return parser.parse_args()


def get_successfully_extracted_images(
    extraction_bucket: str, image_path_prefix: str
) -> List[str]:
    """Get a list of successfully extracted image UUIDs from bucket and prefix

    Args:
        extraction_bucket (str): bucket containing extracted images
        image_path_prefix (str): prefix of image path

    Returns:
        List[str]: List of video UUIDs that have successful extraction
    """
    successful_video_uuids = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for f_name in glob_from_bucket(
            extraction_bucket,
            prefix=image_path_prefix,
            extensions=("json"),
        ):
            if "status.json" not in f_name:
                continue
            status_local_path = f"{tmpdir}/status.json"
            download_to_file(extraction_bucket, f_name, status_local_path)
            with open(status_local_path, encoding="UTF-8") as status_data:
                status = json.loads(status_data.read())
                if status["exit_status"] < 0:
                    continue
            relative_path = f_name.replace(f"{image_path_prefix}/", "")
            video_uuid = relative_path.replace("/status.json", "")
            successful_video_uuids.append(video_uuid)
    return successful_video_uuids


@sematic.func(resource_requirements=CPU_1CORE_4GB, standalone=True)
def get_images_to_extract(
    bucket: str,
    prefix: str,
    label_video_uuids: List[str],
) -> List[str]:
    """Get list of video UUIDs that need extraction

    Args:
        bucket (str): Bucket containing images
        prefix (str): prefix to path to store images
        label_video_uuids (List[str]): List of newly extracted images

    Returns:
        List[str]: List of video UUIDs that need to be extracted
    """
    video_uuids_with_good_images = get_successfully_extracted_images(
        bucket, prefix
    )
    video_uuids_to_extract_images = list(
        set(label_video_uuids) - set(video_uuids_with_good_images)
    )
    logger.info(
        f"get_images_to_extract found {len(video_uuids_to_extract_images)} good labeled images"
    )
    return video_uuids_to_extract_images
