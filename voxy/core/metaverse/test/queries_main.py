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

import os

from loguru import logger

from core.metaverse.metaverse import Metaverse
from core.structs.dataset import DatasetFormat
from core.structs.task import TaskPurpose

# Line too long:
# trunk-ignore-all(pylint/C0301)


def main():
    os.environ["METAVERSE_ENVIRONMENT"] = "INTERNAL"
    metaverse = Metaverse()
    insert_task(metaverse)
    insert_datapool(metaverse)
    query_video_path(metaverse)
    insert_logset(metaverse)
    insert_camera_config(metaverse)
    insert_task(metaverse)
    insert_logset(metaverse)
    insert_camera(metaverse)
    insert_video(metaverse)
    insert_model(metaverse)
    query_camera(metaverse)
    insert_image_logset(metaverse)
    insert_data_collection_logset(metaverse)
    insert_video_logset(metaverse)
    query_video_logset(metaverse)
    query_image_logset(metaverse)
    query_task(metaverse)
    insert_task(metaverse)
    insert_labeling_project(metaverse)
    query_labeling_project(metaverse)
    update_labeling_project(metaverse)
    metaverse.close()


def query_task(metaverse) -> None:
    """Queries task in metaverse

    Args:
        metaverse (Metaverse): metaverse to run query in
    """
    result = metaverse.schema.execute(
        '{task(purpose: "DOOR_STATE", camera_uuid: "americold/ontario/0001/cha") { uuid, purpose, camera_ref { uuid } }}'
    )
    logger.info(result)


def insert_labeling_project(metaverse):
    """Inserts labeling project in metaverse

    Args:
        metaverse (Metaverse): metaverse to run query in
    """
    result = metaverse.schema.execute(
        'mutation{label_project_create(labeling_tool:"scale", name:"video_playback_annotation")'
        "{label_project { name description } success error}}"
    )
    # Expected output
    # ExecutionResult(data={'label_project_create': {'label_project': {'name': 'door_classification',
    # 'description': ''}, 'success': True, 'error': None}}, errors=None)
    logger.info(result)


def query_labeling_project(metaverse):
    """Queries labeling project in metaverse

    Args:
        metaverse (Metaverse): metaverse to run query in
    """
    result = metaverse.schema.execute(
        '{label_project(labeling_tool:"scale", name:"video_playback_annotation") '
        "{ name description last_checked_timestamp }}"
    )
    # Expected output
    # ExecutionResult(data={'label_project': {'name': 'video_playback_annotation', 'description': '',
    # 'last_checked_timestamp': '2021-08-10T00:00:00+00:00'}}, errors=None)
    logger.info(result)


def update_labeling_project(metaverse):
    """Updates labeling project in metaverse

    Args:
        metaverse (Metaverse): metaverse to run query in
    """
    result = metaverse.schema.execute(
        'mutation { label_project_update(labeling_tool:"scale", name:"video_playback_annotation", '
        'updated_time: "2021-08-10T00:00:00.000Z") { label_project { name description '
        "last_checked_timestamp } success error } }"
    )
    # Expected output
    # ExecutionResult(data={'label_project_update': {'label_project': {'name': 'video_playback_annotation',
    # 'description': '', 'last_checked_timestamp': '2021-08-10T00:00:00+00:00'}, 'success': True, 'error':
    # None}}, errors=None)
    logger.info(result)


def insert_task(metaverse):
    """Inserts task in metaverse

    Args:
        metaverse (Metaverse): metaverse to run query in
    """
    result = metaverse.schema.execute(
        f'mutation {{task_create(purpose:"{TaskPurpose.DOOR_STATE.name}", camera_uuids: ["americold/ontario/0003/cha", "americold/ontario/0005/cha"]) {{task {{ purpose }} success error}} }}'
    )
    logger.info(result)


def insert_datapool(metaverse):
    """Inserts task in metaverse

    Args:
        metaverse (Metaverse): metaverse to run query in
    """

    # these are test queries:
    # trunk-ignore-all(semgrep)
    query = (
        '{task_from_cameras(purpose:"DOOR_STATE", camera_uuids: '
        '["americold/ontario/0005/cha"]) {  uuid, purpose, service_ref { uuid, category} } }'
    )
    result = metaverse.schema.execute(query)
    logger.info(result)
    if result.data and result.data["task_from_cameras"] is not None:
        task_uuid = result.data["task_from_cameras"]["uuid"]
        logger.info(f"Task UUID: {task_uuid}")
        if len(result.data["task_from_cameras"]["service_ref"]) < 1:
            logger.error(
                f"Could not find a service attached to this task: {task_uuid}"
            )
            return
        service_uuid = result.data["task_from_cameras"]["service_ref"][0][
            "uuid"
        ]
        # try inserting datapool
        result = metaverse.schema.execute(
            f'mutation {{datapool_create(service_uuid:"{service_uuid}", name: "foo", url: "https://supercoolthing.com/", lightly_uuid:"alkjskldjfljw23", metadata:"{{}}", lightly_config: "{{}}") {{datapool {{ name, version }} }} }}'
        )
        logger.info(result)
    else:
        logger.error(f"Getting the task query failed: {result}")


def insert_image_logset(metaverse) -> None:
    """Create an image logset. If the name is same, the version is automatically bumped.

    Args:
        metaverse (Metaverse): metaverse to run query in
    """
    result = metaverse = metaverse.schema.execute(
        'mutation {image_logset_create(name:"test", images: [{name:"image1"}]) {image_logset { name, version }} }'
    )
    logger.info(result)


def insert_model(metaverse):
    """Tests the model/dataset registry apis"""
    # first make a logset
    result = metaverse.schema.execute(
        'mutation {data_collection_logset_create(name:"video_logset_5", data_collections: [{data_collection_uuid:"video"}]) {data_collection_logset { uuid, name, version }} }'
    )
    logger.info(result)
    # get the uuid
    uuid = result.data["data_collection_logset_create"][
        "data_collection_logset"
    ]["uuid"]
    result = metaverse.schema.execute(
        f'mutation {{dataset_create(format: "{DatasetFormat.names()[0]}", config: "{{}}", path:"s3://to/somewhere", git_version: "version", metadata:"{{}}", logset_uuid:"{uuid}") {{dataset {{ uuid }}}} }}'
    )
    logger.info(result)
    uuid = result.data["dataset_create"]["dataset"]["uuid"]
    result = metaverse.schema.execute(
        f'mutation {{model_create(name:"SafetyVest", config: "{{}}", metadata: "{{}}", path:"s3://to/somewhere", run_links: "[]", dataset_uuid:"{uuid}") {{ model {{ uuid }}}} }}'
    )
    logger.info(result)
    model_uuid = result.data["model_create"]["model"]["uuid"]

    # first create a task
    result = metaverse.schema.execute(
        f'mutation {{task_create(purpose:"{TaskPurpose.PPE_SAFETY_VEST.name}", camera_uuids: ["americold/ontario/0003/cha", "americold/ontario/0005/cha"]) {{task {{ uuid }} error }} }}'
    )
    logger.info(result)
    if result.data["task_create"]["task"] is not None:
        task_uuid = result.data["task_create"]["task"]["uuid"]
        result = metaverse.schema.execute(
            f'mutation {{service_create(task_uuid:"{task_uuid}", metadata: "{{}}", category: "CLASSIFICATION") {{ service {{ uuid }}}} }}'
        )
        logger.info(result)
        service_uuid = result.data["service_create"]["service"]["uuid"]
        # finally add the model to the service
        result = metaverse.schema.execute(
            f'mutation {{service_add_models(model_uuids:["{model_uuid}"], service_uuid: "{service_uuid}") {{ success }} }}'
        )
        logger.info(result)


def insert_video_logset(metaverse):
    """Create a video logset. If the name is same, the version is automatically bumped."""
    result = metaverse.schema.execute(
        'mutation {video_logset_create(name:"test", videos: [{video_uuid:"video"}]) {video_logset { name, version }} }'
    )
    logger.info(result)


def insert_data_collection_logset(metaverse: Metaverse):
    """
    Create a data collection logset. If the name is same, the version is automatically bumped.

    Args:
        metaverse (Metaverse): the current metaverse test instance
    """
    result = metaverse.schema.execute(
        'mutation {data_collection_logset_create(name:"test", data_collections: [{data_collection_uuid:"the-best-uuid-ever"}]) {data_collection_logset { name, version }} }'
    )
    logger.info(result)


def query_video_logset(metaverse):
    result = metaverse.schema.execute(
        ' {  video_logset(name: "test", version:1) { name } }'
    )
    logger.info(result)


def query_image_logset(metaverse):
    result = metaverse.schema.execute(
        ' {  image_logset(name: "test", version:1) { name } }'
    )
    logger.info(result)


def query_video_path(metaverse):
    result = metaverse.schema.execute(
        ' {  video_path_contains(path: "ontario") { uuid } }'
    )
    logger.info(result)


def insert_logset(metaverse):
    result = metaverse.schema.execute(
        'mutation { logset_create(name: "test", videos: [ { video: "39777e19-e197-4ed2-aa7d-071a5f7a40ca", violation_version: "v1", labels_version: "v1" } ]) { logset { name } }}'
    )
    logger.info(result)


def query_camera(metaverse):
    result = metaverse.schema.execute(
        ' {  camera(organization: "uscold", location: "laredo", zone: "room_f1", channel_name: "cha") { uuid, organization,location, camera_config_ref{actionable_regions{polygon}} } } '
    )
    logger.info(result)


def insert_camera(metaverse):
    result = metaverse.schema.execute(
        'mutation { camera_create(uuid: "some_uuid", organization: "uscold", location: \
        "laredo", zone: "room_f1", channel_name: "cha", kinesis_url: "", is_active: TRUE) \
        { camera { uuid } }}'
    )
    logger.info(result)


def insert_camera_config(metaverse):
    """Insert a camera config.

    Args:
        metaverse (Metaverse): metaverse instance
    """
    result = metaverse.schema.execute(
        'mutation { camera_config_create(organization: "uscold", location: "laredo", zone: "room_f1", '
        'channel_name: "cha", doors: [], driving_areas: [], actionable_regions: '
        "[{polygon: [[0.0, 1901.99], [533.12, 763.72], [918.66, 752.91], [1520.0, 1790.94], "
        "[1520.0, 2684.24], [0.0, 2684.24]]}], intersections: [{polygon: [[601.77, 1203.5], "
        "[887.26, 1203.5], [935.58, 1396.75], [961.93, 1664.67], [531.5, 1686.64], [562.24, 1370.4], "
        "[606.16, 1207.89]]}], end_of_aisles: [{points: [[13.2, 2609.0], [1515.3, 2569.5]]}], "
        "no_pedestrian_zones: [] , motion_detection_zones: [], no_obstruction_regions: []) { camera_config { uuid } } } "
    )
    logger.info(result)


def insert_video(metaverse):
    result = metaverse.schema.execute(
        'mutation { video_create(camera_uuid: "fake_organization/fake_location/fake_zone/fake_channel", path:"someotherpath", frames: [{ actors: [{category:"ca"}]}]) { video{uuid} }}'
    )
    logger.info(result)


if __name__ == "__main__":
    main()
