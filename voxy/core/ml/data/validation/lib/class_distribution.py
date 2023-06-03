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

import copy
import os
from typing import List

import sematic

from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.metaverse.utils.door_dataset_utils import (
    get_door_distribution_data_for_camera,
)
from core.structs.actor import DoorState
from core.utils.logging.slack.get_slack_webhooks import (
    get_perception_verbose_sync_webhook,
)
from core.utils.logging.slack.synchronous_webhook_wrapper import (
    SynchronousWebhookWrapper,
)


def create_track_uuid_state_map(door_list: list) -> dict:
    """
    Takes result from metaverse query and creates a map of door
    track uuids to number of classes that is labeled

    Args:
        door_list (list): list of doors dictionary results from metaverse
        containing the door_state and the track_uuid

    Returns:
        Dictionary of doors in video mapping to its class distribution
    """
    track_uuid_state_map = {}
    for door in door_list:
        if (
            door.get("door_state")
            and door.get("track_uuid")
            and door.get("slice")
        ):
            state = door["door_state"]
            uuid = door["track_uuid"]
            data_slice = door["slice"]
            if uuid not in track_uuid_state_map:
                door_state_map = {state.name: 0 for state in DoorState}
                slice_map = {
                    "train": copy.deepcopy(door_state_map),
                    "test": copy.deepcopy(door_state_map),
                }
                track_uuid_state_map[uuid] = slice_map
            uuid_slice_states = track_uuid_state_map[uuid][data_slice]
            uuid_slice_states[state] += 1
    return track_uuid_state_map


def notify_door_state_distribution(
    camera_uuid: str, track_uuid_state_map: dict
) -> None:
    """
    Sends slack notification regarding door_state distribution

    Args:
        camera_uuid (str): camera uuid of door query
        track_uuid_state_map (dict): dictionary of state count per door track uuid
    """
    perception_webhook_notifier = SynchronousWebhookWrapper(
        get_perception_verbose_sync_webhook()
    )
    for track_uuid, slices in track_uuid_state_map.items():
        notification_block = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Door Class Distribution, Camera: {camera_uuid}, Door: {track_uuid}",
                },
            }
        ]
        slice_fields = []
        for training_slice, class_distribution in slices.items():
            slice_message = f"*{training_slice.upper()} VoxelDataset:*\n"
            for door_state, count in class_distribution.items():
                slice_message += f"{door_state}: {count}\n"
            slice_fields.append(
                {
                    "type": "mrkdwn",
                    "text": f"{slice_message}",
                },
            )
        notification_block.append(
            {
                "type": "section",
                "fields": slice_fields,
            }
        )
        perception_webhook_notifier.post_message_block(notification_block)


def door_class_distribution(camera_uuid: str) -> None:
    """
    Gets distribution of door classes per camera uuid

    Args:
        camera_uuid (str): camera_uuid which we want to get distribution of door classes
    """
    doors = get_door_distribution_data_for_camera(camera_uuid)
    door_state_map = create_track_uuid_state_map(doors)
    notify_door_state_distribution(camera_uuid, door_state_map)


@sematic.func(resource_requirements=CPU_1CORE_4GB, standalone=True)
def notify_class_distribution(
    video_uuids: List[str], metaverse_environment: str, project_name: str
) -> bool:
    """
    Gets distribution of door classes per camera uuid specified contained within the list of videos
    Args:
        video_uuids (List[str]): list of video uuids
        metaverse_environment (str): metaverse environment
        project_name (str): name of the project in scale
    Returns:
        bool: distribution extraction status boolean
    """
    if project_name == "door_state_classification":
        os.environ["METAVERSE_ENVIRONMENT"] = metaverse_environment
        camera_uuids = {
            ("/").join(video.split("/")[0:4]) for video in video_uuids
        }
        for camera_uuid in camera_uuids:
            door_class_distribution(camera_uuid)
        return True
    return False
