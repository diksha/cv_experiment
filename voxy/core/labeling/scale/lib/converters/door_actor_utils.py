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

import uuid
from typing import Any

from scaleapi.tasks import Task

from core.structs.actor import DoorState
from core.structs.video import Video


def _generate_door_actor(task: Task) -> dict:
    """
    Generates a door actor given a labeled scale task

    Args:
        task (Task): scale api task

    Returns:
        door_actor (dict): dictionary representation of door actor
    """
    door_object = {}
    door_object["category"] = "DOOR"
    door_object["track_uuid"] = task.metadata["track_uuid"]
    door_object["door_orientation"] = task.metadata["door_orientation"]
    door_state = task.response["annotations"]["door_state"][0]
    door_object["door_state"] = "UNKNOWN"
    if DoorState.__members__.get(door_state):
        door_object["door_state"] = door_state
    door_object["door_type"] = task.metadata["door_type"]
    door_object["polygon"] = task.metadata["door_polygon"]
    door_object["uuid"] = str(uuid.uuid4())
    return door_object


def generate_consumable_labels_for_doors(
    video_uuid: str, video_task_list: list
) -> Video:
    """
    Generates and uploads video consumable labels for doors to gcs

    Args:
        video_uuid (str): voxel formatted video uuid for labels
        video_task_list (list): list of scale api Tasks for the video

    Returns:
        upload_success (bool): bool of whether or not the upload to gcs was successful
    """
    timestamp_ms_frame_struct_map = {}
    for task in sorted(
        video_task_list, key=lambda x: x.metadata["timestamp_ms"]
    ):
        local_frame_struct = {}
        timestamp_ms = float(task.metadata["timestamp_ms"])
        if timestamp_ms not in timestamp_ms_frame_struct_map:
            local_frame_struct["frame_number"] = None
            local_frame_struct["frame_width"] = task.metadata["width"]
            local_frame_struct["frame_height"] = task.metadata["height"]
            local_frame_struct["relative_timestamp_ms"] = timestamp_ms
            local_frame_struct["relative_timestamp_s"] = float(
                timestamp_ms / 1000
            )
            local_frame_struct["actors"] = []
            timestamp_ms_frame_struct_map[timestamp_ms] = local_frame_struct
        else:
            local_frame_struct = timestamp_ms_frame_struct_map[timestamp_ms]
        current_door = _generate_door_actor(task)
        door_exists_in_frame = False
        for door in local_frame_struct["actors"]:
            if door["track_uuid"] == current_door["track_uuid"]:
                door = current_door
                door_exists_in_frame = True
                break
        if not door_exists_in_frame:
            local_frame_struct["actors"].append(current_door)
    frame_struct_list = []
    for _, frame_struct in timestamp_ms_frame_struct_map.items():
        frame_struct_list.append(frame_struct)
    consumable_labels: dict[str, Any] = {}
    consumable_labels["uuid"] = video_uuid
    consumable_labels["parent_uuid"] = None
    consumable_labels["root_uuid"] = None
    consumable_labels["frames"] = frame_struct_list
    return Video.from_dict(consumable_labels)
