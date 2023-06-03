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
from time import time
from typing import List

import sematic
from loguru import logger

from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.labeling.label_store.label_reader import LabelReader
from core.ml.data.extraction.extractors.image_extractor import ImageExtractor
from core.ml.data.extraction.writers.image_writer import ImageWriter
from core.ml.data.extraction.writers.status_writer import StatusWriter
from core.utils.video_reader import S3VideoReader, S3VideoReaderInput


def parse_args() -> argparse.Namespace:
    """Parse commandline arguments

    Returns:
        argparse.Namespace: parsed commandline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bucket",
        type=str,
        default="voxel-datasets",
        help="Bucket to extract images to",
    )
    parser.add_argument(
        "-r",
        "--relative_path",
        type=str,
        required=True,
        help="Relative path from bucket to store images in",
    )
    parser.add_argument(
        "-v",
        "--video_uuid",
        type=str,
        required=True,
        help="Video UUID to extract images from",
    )

    return parser.parse_args()


def extract_video_images(bucket: str, relative_path: str, video_uuid: str):
    """Extract images from a single video

    Args:
        bucket (str): cloud storage bucket
        relative_path (str): cloud storage relative path
        video_uuid (str): UUID of data collection to extract
    """
    video_reader_input = S3VideoReaderInput(
        video_path_without_extension=video_uuid
    )
    video_reader = S3VideoReader(video_reader_input)
    image_writer = ImageWriter(
        bucket=bucket,
        relative_path=relative_path,
    )
    status_writer = StatusWriter()
    label_reader = LabelReader()
    labels = json.loads(label_reader.read(video_uuid))
    frames_tms_to_extract = {
        frame["relative_timestamp_ms"] for frame in labels["frames"]
    }
    image_extractor = ImageExtractor(
        video_reader,
        image_writer,
        status_writer,
        frames_tms_to_extract,
    )
    image_extractor.extract_data()


@sematic.func(resource_requirements=CPU_1CORE_4GB, standalone=True)
def extract_images(
    bucket: str, relative_path: str, video_uuids: List[str]
) -> List[str]:
    """Sematic pipeline func to extract images for YOLO

    Args:
        bucket (str): cloud storage bucket
        relative_path (str): cloud storage relative path
        video_uuids (List[str]): UUIDs of videos to extract images from

    Returns:
        List[str]: video UUIDs extracted
    """
    extracted_uuids = []
    start_time = time()
    for video_uuid in video_uuids:
        logger.info(f"extract_image({video_uuid})")
        extract_video_images(bucket, relative_path, video_uuid)
        extracted_uuids.append(video_uuid)
    elapsed = time() - start_time
    logger.info(
        f"extract_images extracted {len(extracted_uuids)} images in {elapsed}s"
    )
    return extracted_uuids


if __name__ == "__main__":
    args = parse_args()
    extract_video_images(args.bucket, args.relative_path, args.video_uuid)
