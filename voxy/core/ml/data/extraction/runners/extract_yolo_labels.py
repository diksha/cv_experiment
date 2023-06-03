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
import os
from typing import List

import sematic
from loguru import logger

from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.labeling.label_store.label_reader import LabelReader
from core.ml.data.extraction.extractors.label_extractor import LabelExtractor
from core.ml.data.extraction.label_generation.yolo_label_generator import (
    YoloLabelGenerator,
)
from core.ml.data.extraction.writers.label_writer import LabelWriter
from core.ml.data.extraction.writers.status_writer import StatusWriter
from core.structs.actor import ActorCategory

YOLO_DIR = "yolo"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--bucket",
        type=str,
        default="voxel-datasets",
        help="Bucket to save annotations to",
    )
    parser.add_argument(
        "-r",
        "--base_relative_path",
        type=str,
        required=True,
        help="base relative path in bucket to save annotations to",
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name for labels",
    )
    parser.add_argument(
        "-l",
        "--label_directory",
        type=str,
        required=True,
        help="Label directory name",
    )
    parser.add_argument(
        "-a",
        "--actors_to_keep",
        type=str,
        nargs="+",
        required=True,
        help="Ordered actors to extract annotations for",
    )
    parser.add_argument(
        "-v",
        "--video_uuid",
        type=str,
        required=True,
        help="Video UUID to extract labels for",
    )

    return parser.parse_args()


def extract_yolo_label_for_video(
    label_generator: YoloLabelGenerator,
    base_relative_path: str,
    dataset_name: str,
    label_directory: str,
    bucket: str,
    video_uuid: str,
):
    """Extract a YOLO labels for a single video

    Args:
        label_generator (YoloLabelGenerator): label generator to use
        base_relative_path (str): relative path of output
        dataset_name (str): name for this output dataset
        label_directory (str): subdirectory of dataset_name to store labels in
        bucket (str): bucket to store output
        video_uuid (str): UUID of video to extract label of
    """
    label_reader = LabelReader()
    relative_path = os.path.join(
        base_relative_path,
        os.path.join(YOLO_DIR, os.path.join(dataset_name, label_directory)),
    )

    label_writer = LabelWriter(
        label_generator,
        bucket,
        relative_path,
    )
    status_writer = StatusWriter()
    label_extractor = LabelExtractor(
        video_uuid,
        label_reader,
        label_writer,
        status_writer,
    )
    label_extractor.extract_data()


@sematic.func(resource_requirements=CPU_1CORE_4GB, standalone=True)
def extract_yolo_labels(
    actors_to_keep: List[str],
    base_relative_path: str,
    dataset_name: str,
    label_directory: str,
    bucket: str,
    video_uuids: List[str],
) -> List[str]:
    """Extract YOLO labels for given data collection UUIDs

    Args:
        actors_to_keep (List[str]): List of actor categories to keep
        base_relative_path (str): Relative base path for data collection
        dataset_name (str): Name of dataset
        label_directory (str): Directory for output of labels
        bucket (str): Bucket for output
        video_uuids (List[str]): UUIDs of videos to extract labels

    Returns:
        List[str]: UUIDs with extracted labels
    """
    label_generator = YoloLabelGenerator(
        actors_to_keep=[ActorCategory[actor] for actor in actors_to_keep]
    )

    for video_uuid in video_uuids:
        extract_yolo_label_for_video(
            label_generator,
            base_relative_path,
            dataset_name,
            label_directory,
            bucket,
            video_uuid,
        )
    logger.info(f"extract_yolo_labels found {len(video_uuids)} labels")
    return video_uuids


def main():
    """Main function to extract YOLO labels"""
    args = parse_args()

    label_generator = YoloLabelGenerator(
        actors_to_keep=[ActorCategory[actor] for actor in args.actors_to_keep]
    )

    extract_yolo_label_for_video(
        label_generator,
        args.base_relative_path,
        args.dataset_name,
        args.label_directory,
        args.bucket,
        args.video_uuid,
    )


if __name__ == "__main__":
    main()
