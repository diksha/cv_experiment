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
from loguru import logger
from scaleapi.tasks import TaskStatus

from core.labeling.scale.lib.scale_batch_wrapper import ScaleBatchWrapper
from core.labeling.scale.lib.scale_task_wrapper import ScaleTaskWrapper


def _video_uuid_to_tasks_map(task_list: list) -> dict:
    """
    Create a mapping associating videos to their respective tasks

    Args:
        task_list (list): list of all tasks in a batch

    Returns:
        video_task_map (dict): mapping of video name to respective labeling tasks
    """
    video_task_map = {}
    for task in task_list:
        video_uuid = task.metadata["video_uuid"]
        if video_uuid not in video_task_map:
            video_task_map[video_uuid] = []
        video_task_map[video_uuid].append(task)
    return video_task_map


def get_completed_batch_video_map(
    project_name, completed_after, completed_before, credentials_arn
) -> dict:
    """Get completed batch to list of videos map

    Args:
        project_name (str): name of the project
        completed_after (str): batch completed after date
        completed_before (str): batch completed before date
        credentials_arn (str): scale credentials arn

    Returns:
        dict: batch_name to list of videos and video tasks map.
    """
    task_list = ScaleTaskWrapper(credentials_arn).get_active_tasks(
        project_name=project_name,
        updated_after=completed_after,
        updated_before=completed_before,
        status=TaskStatus("completed"),
    )
    completed_batches = ScaleBatchWrapper(
        credentials_arn
    ).get_completed_batches(task_list)
    batch_log = ", ".join(completed_batches)
    logger.info(f"Converting {project_name} batches: {batch_log}")
    batch_map = {}
    for batch_name in completed_batches:
        batch_tasks = ScaleTaskWrapper(credentials_arn).get_active_tasks(
            project=project_name,
            batch_name=batch_name,
            status="completed",
        )
        video_task_map = _video_uuid_to_tasks_map(batch_tasks)
        batch_map[batch_name] = video_task_map
    return batch_map
