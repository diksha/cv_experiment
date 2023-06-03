#
# Copyright 2020-2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import os
import tempfile
from typing import Any, Dict, List, Optional

import cv2

from core.labeling.scale.lib.converters.polygon_utils import (
    get_polygon_vertices,
)
from core.structs.actor import HeadCoveringType
from core.structs.video import Video
from core.utils.aws_utils import download_to_file, get_bucket_path_from_s3_uri


def generate_consumable_labels_for_ppe_hat(
    video_uuid: str, video_task_list: list
) -> Video:
    """
    Generates and uploads video consumable labels for safety vest to aws

    Args:
        video_uuid (str): voxel formatted video uuid for labels
        video_task_list (list): list of scale api Tasks for the video

    Returns:
        upload_success (bool): bool of whether or not the upload to aws was successful
    """

    def get_head_covering_type_for_track_uuid(
        head_covering_track_uuid: str, annotations: List[Dict[str, object]]
    ) -> Optional[str]:
        """Scans the actors in the image to find the selected track
        uuid and returns the head covering type. If the track_uuid
        is not a head_covering actor, returns None
        Args:
            head_covering_track_uuid (str): track uuid of head covering actor
            annotations (List[Dict[str, object]]): list of annotations for the image
        Returns:
            HeadCoveringType: head covering type of the actor
        """
        for actor in annotations:
            if actor["uuid"] == head_covering_track_uuid:
                return HeadCoveringType[actor["label"]].name
        return None

    frame_struct_list = []
    for task in video_task_list:
        bucket, path = get_bucket_path_from_s3_uri(task.params["attachment"])
        with tempfile.NamedTemporaryFile() as temp:
            download_to_file(bucket, path, temp.name)
            image = cv2.imread(temp.name)
            height, width, _ = image.shape
        local_frame_struct = {}
        local_frame_struct["relative_path"] = task.metadata["relative_path"]
        local_frame_struct["frame_number"] = None
        local_frame_struct["frame_width"] = width
        local_frame_struct["frame_height"] = height
        local_frame_struct["actors"] = []
        local_frame_struct["relative_image_path"] = os.path.relpath(
            path, video_uuid
        )
        for annotation in task.response["annotations"]:
            label_attributes = {}
            label_attributes["occluded_degree"] = None
            label_attributes["truncated"] = None
            label_attributes["category"] = annotation["label"]

            polygon = {}
            polygon["vertices"] = get_polygon_vertices(annotation)
            label_attributes["polygon"] = polygon
            label_attributes["uuid"] = annotation["uuid"]

            for key_attribute in annotation["attributes"].keys():
                if key_attribute == "head_covered_state":
                    label_attributes[
                        "head_covering_type"
                    ] = get_head_covering_type_for_track_uuid(
                        annotation["attributes"][key_attribute],
                        task.response["annotations"],
                    )
                    label_attributes["is_wearing_hard_hat"] = (
                        HeadCoveringType[
                            label_attributes["head_covering_type"]
                        ]
                        == HeadCoveringType.HARD_HAT
                    )
                else:
                    attribute_val = annotation["attributes"][key_attribute]
                    label_attributes[key_attribute] = (
                        attribute_val.lower() == "true"
                        if (attribute_val.lower() in ["true", "false"])
                        else attribute_val
                    )
            local_frame_struct["actors"].append(label_attributes)
        frame_struct_list.append(local_frame_struct)
    consumable_labels: dict[str, Any] = {}
    consumable_labels["uuid"] = video_uuid
    consumable_labels["parent_uuid"] = None
    consumable_labels["root_uuid"] = None
    consumable_labels["frames"] = frame_struct_list

    return Video.from_dict(consumable_labels)
