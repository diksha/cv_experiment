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
from ast import literal_eval

from core.incidents.utils import CameraConfig
from core.metaverse.metaverse import Metaverse
from core.structs.actor import ActorCategory, get_track_uuid


def get_door_distribution_data_for_camera(camera_uuid: str) -> list:
    """
    Queries metaverse to get list of doors given a camera uuid

    Args:
        camera_uuid (str): camera uuid where we want to get lightly doors

    Returns:
        list of doors from metaverse in camera_uuid with door_state, track_uuid, and training slice information

    Raises:
        RuntimeError: metaverse query fails
    """
    metaverse = Metaverse()
    # TODO: Refactor to utility function for reusability
    camera_config = CameraConfig(camera_uuid, -1, -1)
    door_uuids = json.dumps(
        [
            get_track_uuid(camera_uuid, str(door.door_id), ActorCategory.DOOR)
            for door in camera_config.doors
        ]
    )
    query = """query get_actors_from_data_collection_slice(
        $track_uuids: [String],
        $data_collection_slice: String
    ) {
        actors_from_data_collection_slice(
            track_uuids: $track_uuids,
            data_collection_slice: $data_collection_slice
        ) {
            door_state, track_uuid
        }
    }
    """
    train_q_vars = {
        "track_uuids": literal_eval(door_uuids),
        "data_collection_slice": "train",
    }
    test_q_vars = {
        "track_uuids": literal_eval(door_uuids),
        "data_collection_slice": "test",
    }
    train_results = metaverse.schema.execute(query, variables=train_q_vars)
    test_results = metaverse.schema.execute(query, variables=test_q_vars)
    if train_results.errors:
        raise RuntimeError(
            (
                f"Failed to query training doors for camera"
                f" {camera_uuid}, error, {train_results.errors}"
            )
        )
    if test_results.errors:
        raise RuntimeError(
            (
                f"Failed to query testing doors for camera"
                f" {camera_uuid}, error, {test_results.errors}"
            )
        )
    for train_data in train_results.data["actors_from_data_collection_slice"]:
        train_data["slice"] = "train"
    for test_data in test_results.data["actors_from_data_collection_slice"]:
        test_data["slice"] = "test"
    results = (
        train_results.data["actors_from_data_collection_slice"]
        + test_results.data["actors_from_data_collection_slice"]
    )
    return results
