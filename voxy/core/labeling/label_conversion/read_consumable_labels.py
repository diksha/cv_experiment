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
import json
from typing import Any, List, Tuple

from loguru import logger

from core.metaverse.api.data_collection_queries import (
    add_data_collection_labels,
    get_or_create_camera_uuid,
)
from core.structs.data_collection import DataCollectionType
from core.structs.video import Video


class LabelIngestionError(RuntimeError):
    pass


def remove_none(obj):
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_none(x) for x in obj if x is not None)
    if isinstance(obj, dict):
        return type(obj)(
            (remove_none(k), remove_none(v))
            for k, v in obj.items()
            if k is not None and v is not None
        )
    return obj


class MyDict(dict):
    def __repr__(self):
        return "{" + ", ".join([f"{k}: {v!r}" for k, v in self.items()]) + "}"


def hook_fn(value: List[Tuple[str, Any]]):
    return MyDict(value)


def curate_data_collection_metadata(data_metadata: dict) -> str:
    """Change data metadata to metaverse format

    Args:
        data_metadata (dict): data metadata

    Returns:
        str: data metadata in string format
    """
    data_metadata = remove_none(data_metadata)
    curated_data_metadata = str(
        json.loads(json.dumps(data_metadata), object_pairs_hook=hook_fn)
    )
    curated_data_metadata = (
        curated_data_metadata.replace("'", '"')
        .replace("True", "true")
        .replace("False", "false")
    )
    return curated_data_metadata


def convert_xml_to_metadata(
    video_uuid: str, video: Video, incidents: List
) -> str:
    """Curate xml to data metadata format

    Args:
        video_uuid (str): video uuid
        video (Video): video struct contains labels
        incidents (List): incidents for the videos

    Returns:
        str: metadata in string to add to metaverse
    """
    camera_uuid = ("/").join(video_uuid.split("/")[0:4])
    camera_uuid = get_or_create_camera_uuid(camera_uuid)
    video_dict = video.to_dict()
    xml_metadata = {}
    frames = []
    for frame in video_dict["frames"]:
        actors = []
        for actor in frame["actors"]:
            del actor["uuid"]
            actors.append(actor)
        frame["actors"] = actors
        frames.append(frame)
    xml_metadata["frames"] = frames
    xml_metadata["voxel_uuid"] = video_dict["uuid"]
    if incidents is not None:
        xml_metadata["violations"] = [
            {"version": "v1", "violations": incidents}
        ]
    return curate_data_collection_metadata(xml_metadata)[1:-1]


def read(
    video: Video,
    video_uuid: str,
    incidents: List,
    label_metadata: dict,
):
    """Reads the labels and stores it in metaverse

    Args:
        video(Video): gcs file that contains the labels
        video_uuid(str): uuid of the video
        incidents(List): for scenarios, incidents in the video
        label_metadata(dict): labeling metadta

    Raises:
        LabelIngestionError: problem ingesting labels
    """
    label_argument = str(
        json.loads(
            json.dumps(label_metadata.to_dict()), object_pairs_hook=hook_fn
        )
    ).replace("'", '"')
    xml_metadata = convert_xml_to_metadata(video_uuid, video, incidents)

    data_collection_result = add_data_collection_labels(
        xml_metadata, label_argument, DataCollectionType.VIDEO
    )
    logger.info(f"Datacollection {data_collection_result}")
    if not data_collection_result.data.get("data_collection_update", {}).get(
        "success"
    ):
        raise LabelIngestionError(
            "data_collection_update query not successful!"
        )
    logger.info(
        f"Result of running datacollection query {data_collection_result}"
    )
