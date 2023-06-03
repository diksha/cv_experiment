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
        "--image_bucket",
        metavar="B",
        type=str,
        default="voxel-datasets",
        help="Bucket of extracted data",
    )
    parser.add_argument(
        "--label_bucket",
        metavar="B",
        type=str,
        default="voxel-datasets",
        help="Bucket of extracted data",
    )
    parser.add_argument(
        "--image_relative_path",
        metavar="B",
        type=str,
        required=True,
        help="Relative path to images from image bucket",
    )
    parser.add_argument(
        "--label_relative_path",
        metavar="B",
        type=str,
        required=True,
        help="Relative path to labels from image bucket",
    )
    return parser.parse_args()


def get_video_uuids_with_successful_extraction(
    bucket_name: str, path_prefix: str
) -> List[str]:
    video_uuids = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for f_name in glob_from_bucket(
            bucket_name,
            prefix=path_prefix,
            extensions=("json"),
        ):
            if "status.json" not in f_name:
                continue
            status_local_path = f"{tmpdir}/status.json"
            download_to_file(bucket_name, f_name, status_local_path)
            with open(status_local_path, encoding="UTF-8") as status_data:
                status = json.loads(status_data.read())
                if status["exit_status"] == 0:
                    prefix_removed = f_name.replace(f"{path_prefix}/", "")
                    video_uuid = prefix_removed.replace("/status.json", "")
                    video_uuids.append(video_uuid)
    return video_uuids


@sematic.func(resource_requirements=CPU_1CORE_4GB, standalone=True)
def validate_uuids(
    image_extraction_successful_uuids: List[str],
    label_extraction_successful_uuids: List[str],
    image_bucket: str,
    image_relative_path: str,
) -> List[str]:
    """Get UUIDs with both labels and data

    Args:
        image_extraction_successful_uuids (List[str]): UUIDs of images successfully extracted
            (unused, only for sematic data dependency)
        label_extraction_successful_uuids (List[str]): UUIDs with successful label extraction
        image_bucket (str): bucket storing extracted images
        image_relative_path (str): relative path of extracted images

    Returns:
        List[str]: UUIDs with both labels and data
    """
    image_extraction_successful_uuids = (
        get_video_uuids_with_successful_extraction(
            image_bucket, image_relative_path
        )
    )

    valid_uuids = list(
        set(image_extraction_successful_uuids).intersection(
            set(label_extraction_successful_uuids)
        )
    )
    logger.info(f"validate_uuids found {len(valid_uuids)} valid UUIDs")
    return valid_uuids
