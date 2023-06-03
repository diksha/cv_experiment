#
# Copyright 2020-2021 Voxel Labs, Inc.
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
import os
from enum import Enum
from typing import List

import sematic

from core.common.utils.recursive_namespace import RecursiveSimpleNamespace
from core.infra.cloud.firestore_utils import read_from_output_store
from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.labeling.label_conversion.read_consumable_labels import read
from core.labeling.scale.lib.converters.converter_base import VideoData
from core.structs.video import Video
from core.utils.aws_utils import get_blobs_contents_from_bucket


class LabelMetadata(RecursiveSimpleNamespace):
    """
    Simple wrapper class for label metadata and all it's helper
    methods
    """


class LabelTaxonomy(Enum):
    """Taxonomy for the labels

    Args:
        Enum: enum type
    """

    UNKNOWN = 0
    SAFETY_VEST = 1
    DOOR_STATE_CLASSIFICATION = 2
    VIDEOPLAYBACK = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--video",
        metavar="V",
        type=str,
        help="video uuid to ingest",
        default="",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="source of label",
        default="scale",
    )
    parser.add_argument(
        "-p",
        "--project",
        type=str,
        help="project for label",
        default="door_state_classification",
    )
    parser.add_argument(
        "-t",
        "--taxonomy_version",
        type=str,
        help="sha of taxonomy",
    )
    return parser.parse_args()


def get_label_metadata(source: str, project: str) -> LabelMetadata:
    """Gets label metadata for source and project.

    Note: We are storing some derived attributes here

    Args:
        source (str): labeling tool for labels
        project (str): project that labels are coming from

    Raises:
       RuntimeError: Source and project are incorrect

    Returns:
        LabelMetadata: metadata of the labels
    """
    if source == "scale" and project == "door_state_classification":
        taxonomy = LabelTaxonomy.DOOR_STATE_CLASSIFICATION
    elif source == "scale" and project == "safety_vest_image_annotation":
        taxonomy = LabelTaxonomy.SAFETY_VEST
    elif source == "scale" and project == "video_playback_annotation":
        taxonomy = LabelTaxonomy.VIDEOPLAYBACK
    else:
        raise RuntimeError(
            f"No taxonomy for Source {source} and project {project}"
        )
    label_metadata_dict = {
        "source": source,
        "project_name": project,
        "taxonomy": taxonomy.name,
    }
    return LabelMetadata(**label_metadata_dict)


@sematic.func(resource_requirements=CPU_1CORE_4GB, standalone=True)
def add_consumable_label_batch_to_metaverse(
    videos_data: List[VideoData],
    metaverse_environment: str,
    label_source: str,
    label_project: str,
) -> List[str]:
    """
    Adds videos to metaverse
    Args:
        videos_data (List[VideoData]): video uuid list to ingest to metaverse
        metaverse_environment (str): metaverse environment
        label_source (str): labeling tool for labels
        label_project (str): project that labels are coming from
    Returns:
        List[str]: successfully ingested video_uuids
    """
    os.environ["METAVERSE_ENVIRONMENT"] = metaverse_environment
    label_metadata = get_label_metadata(label_source, label_project)
    for video_data in videos_data:
        label_metdata_video = LabelMetadata(
            **label_metadata.to_dict(),
            taxonomy_version=video_data.metadata["taxonomy_version"],
        )
        for file_bytes in get_blobs_contents_from_bucket(
            "voxel-consumable-labels", prefix="v1/" + video_data.video_uuid
        ):
            video = Video.from_dict(json.loads(file_bytes))
            read(video, video_data.video_uuid, None, label_metdata_video)
    return [video_data.video_uuid for video_data in videos_data]


if __name__ == "__main__":
    args = parse_args()
    if not args.video:
        videos_uuid_main = read_from_output_store(
            "buildkite",
            os.getenv("BUILDKITE_BUILD_ID"),
            "image_label_converter",
        )
    else:
        videos_uuid_main = [args.video]

    label_metadata_main = get_label_metadata(args.source, args.project)
    label_metadata_changed = LabelMetadata(
        **label_metadata_main.to_dict(), taxonomy_version=args.taxonomy_version
    )
    for video_uuid_main in videos_uuid_main:
        for label_file_bytes in get_blobs_contents_from_bucket(
            "voxel-consumable-labels", prefix="v1/" + video_uuid_main
        ):
            video_main = Video.from_dict(json.loads(label_file_bytes))
            read(video_main, video_uuid_main, None, label_metadata_changed)
