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

from typing import Tuple

from core.structs.task import Task, TaskPurpose


def get_door_logset_query(task: Task) -> Tuple[str, str, dict]:
    """Gets logset query for doors.

    Args:
        task (Task): Task to get logset query for

    Raises:
        RuntimeError: If more than 1 camera belongs to task

    Returns:
        Tuple[str, str, dict]: query name and query for generating logset.
    """
    if len(task.camera_uuids) != 1:
        raise RuntimeError(
            f"Only 1 camera uuid per door, found {task.camera_uuids}"
        )

    query = """query get_door_logset($camera_uuids: [String])
    {
        video_with_actor_category_from_camera_uuids(category:"DOOR", camera_uuids: $camera_uuids)
        {
            uuid,
            voxel_uuid,
        }
    }
    """
    return (
        query,
        "video_with_actor_category_from_camera_uuids",
        {"camera_uuids": list(task.camera_uuids)},
    )


def get_safety_vest_logset_query(task: Task) -> Tuple[str, str, dict]:
    """Gets logset query for safety vest.

    Args:
        task (Task): Task to get logset query for

    Returns:
        Tuple[str, str, dict]: query name and query for generating logset.
    """
    query = """query get_safety_vest_logset($camera_uuids: [String])
    {
        video_with_actor_category_from_camera_uuids(category:"PERSON", camera_uuids: $camera_uuids)
        {
            uuid,
            voxel_uuid,
        }
    }
    """
    return (
        query,
        "video_with_actor_category_from_camera_uuids",
        {"camera_uuids": list(task.camera_uuids)},
    )


TASK_LOGSET_QUERY_MAP = {
    TaskPurpose.DOOR_STATE: get_door_logset_query,
    TaskPurpose.PPE_SAFETY_VEST: get_safety_vest_logset_query,
}
