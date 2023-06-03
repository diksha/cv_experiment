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

"""Central location for all datacollection queries.
"""

from graphql.execution import ExecutionResult
from loguru import logger

from core.metaverse.metaverse import Metaverse
from core.structs.data_collection import DataCollectionType


def ingest_data_collection(
    camera_uuid: str,
    data_collection_uuid: str,
    is_test: bool,
    data_collection_type: DataCollectionType,
) -> ExecutionResult:
    """Ingest datacollection node to metaverse

    Args:
        camera_uuid (str): uuid of the camera node
        data_collection_uuid (str): data collection uuid in voxel format
        is_test (bool): whether datacollection is for test
        data_collection_type (DataCollectionType): type of datacollection
    Returns:
        ExecutionResult: result of adding to metaverse.
    """
    path = f"s3://voxel-logs/{data_collection_uuid}.mp4"
    if data_collection_type == DataCollectionType.IMAGE_COLLECTION:
        path = f"s3://voxel-logs/{data_collection_uuid}/"
    query = """mutation createDataCollection(
        $camera_uuid:String,
        $path:String,
        $is_test: Boolean,
        $name:String,
        $voxel_uuid:String,
        $data_collection_type:String,
    ) {
        data_collection_create(
            camera_uuid: $camera_uuid,
            path: $path,
            is_test: $is_test,
            name: $name,
            voxel_uuid: $voxel_uuid,
            data_collection_type: $data_collection_type
        )  {
            data_collection {
                uuid
            },
            success
        }
    }
    """
    qvars = {
        "camera_uuid": camera_uuid,
        "path": path,
        "is_test": is_test,
        "name": data_collection_uuid,
        "voxel_uuid": data_collection_uuid,
        "data_collection_type": data_collection_type.name,
    }
    return Metaverse().schema.execute(query, variables=qvars)


def add_data_collection_labels(
    data_collection_metadata: str,
    label_argument: str,
    data_collection_type: DataCollectionType,
) -> ExecutionResult:
    """Add data_collection labels to metaverse

    Args:
        data_collection_metadata (str): metadata of data_collection
        label_argument (str): label metadata of data_collection
        data_collection_type (DataCollectionType): type of data collection

    Returns:
        ExecutionResult: result of adding to metaverse.
    """
    mutation = (
        f"mutation {{ data_collection_update({data_collection_metadata}, "
        f"label_metadata:{label_argument})"
        "{ data_collection {uuid}, success }}"
    )
    return Metaverse().schema.execute(mutation)


def get_or_create_camera_uuid(camera_uuid: str):
    """Given a file from gcs, create or get camera from metaverse.

    Example filename: v1/americold/ontario/0004/cha/label.json

    Args:
        camera_uuid (str): uuid of the camera

    Raises:
        RuntimeError: If filename not in right format

    Returns:
        str: camera uuid for the data_collection
    """
    query = """
        query camera($uuid:String){
            camera(uuid: $uuid) {
                uuid
            }
        }
    """
    qvars = {"uuid": camera_uuid}
    result = Metaverse().schema.execute(query, variables=qvars)
    if result.data["camera"]:
        return result.data["camera"][0]["uuid"]
    mutation = """
        mutation create_camera
        (
            $organization:String!,
            $location:String!,
            $zone:String!,
            $channel_name:String!,
            $uuid:String,
            $is_active:Boolean
        )
        {
            camera_create
            (
                organization: $organization,
                location: $location,
                zone: $zone,
                channel_name: $channel_name,
                uuid: $uuid,
                is_active: $is_active
            )
            {
                camera
                {
                    uuid
                }
            }
        }
        """
    camera_array = camera_uuid.split("/")
    mvars = {
        "organization": camera_array[0],
        "location": camera_array[1],
        "zone": camera_array[2],
        "channel_name": camera_array[3],
        "uuid": camera_uuid,
        "is_active": True,
    }
    result = Metaverse().schema.execute(mutation, variables=mvars)
    # If multiple jobs try to create camera, some will fail and the else will take care of
    # getting the results from camera node that has already been created by another job.
    if result.data["camera_create"]:
        return result.data["camera_create"]["camera"]["uuid"]
    logger.error(
        f"Failed to get_or_create_camera_uuid,\
        camera_uuid, {camera_uuid},\
        get_camera_query, {query},\
        get_camera_query_vars, {qvars},\
        create_camera_mutation, {mutation},\
        create_camera_mutation_vars, {mvars}"
    )
    raise RuntimeError("Failed get_or_create_camera_uuid")
