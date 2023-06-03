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
"""
Syncs camera config data from CVAT tasks to Voxel artifacts.

CVAT project:
https://cvat.voxelplatform.com/projects/10

Usage:
$ bin/python core/labeling/tools/sync_camera_config.py

https://app.clickup.com/36001183/v/dc/12ancz-8240/12ancz-5860 - How to sync camera_config
"""
import argparse
import json
from ast import literal_eval
from datetime import datetime

import requests
from loguru import logger

from core.execution.utils.graph_config_utils import (
    validate_scale_and_graph_config_incidents,
)
from core.labeling.scale.runners.camera_config_raw_labels import (
    camera_config_task_to_raw_labels,
)
from core.labeling.tools.utils import (
    set_camera,
    validate_camera_config_response,
)
from core.structs.actor import DOOR_TYPE_MAP, DOOR_TYPE_PRIORITY
from core.structs.attributes import Point, Polygon, RectangleXYWH
from core.utils.aws_utils import (
    get_blobs_from_bucket,
    get_secret_from_aws_secret_manager,
)
from core.utils.json_utils import underscore_to_camel

SRC_PREFIX = "camera_config/"
MIN_DOOR_RESOLUTION = 7200
MIN_EXIT_DOOR_RESOLUTION = 3600


def discard_different_door_type(door_types, doors):
    highest_priority_door_type = None
    curated_doors = []
    for door_type in door_types:
        door_type = door_type.lower()
        if (
            not highest_priority_door_type
            or DOOR_TYPE_PRIORITY[DOOR_TYPE_MAP[door_type]]
            > DOOR_TYPE_PRIORITY[DOOR_TYPE_MAP[highest_priority_door_type]]
        ):
            highest_priority_door_type = door_type
    for door in doors:
        if door["type"].lower() == highest_priority_door_type.lower():
            curated_doors.append(door)
        else:
            logger.error(f"Discarding door of type {door['type']}")
    return curated_doors


def get_normalized_points_for_scale(
    vertices, original_height, original_width, min_resolution
) -> list:
    """Gets list of normalized points from an encoded polygon element string.

    Input example:
    "350.66,60.06;580.47,70.60;549.55,251.22;342.23,240.67;350.66,60.06"
    height: 720.0
    width: 1280.0

    Output example:
    [[0.273953125, 0.08341666666], [0.4534921875, 0.09805555555], [0.4293359375, 0.34891666666]
    , [0.2673671875, 0.33426388888], [0.273953125, 0.08341666666]]

    Args:
        vertices (str): Vertices of polygon
        original_height (int): Height of image on which camera config was created.
        original_width (int): Width of image on which camera config was created.
        min_resolution (int): minimum resolution to add to camera config

    Returns:
        list: [List of points, is_polygon_valid]
    """
    normalized_output = []
    output = []
    for vertex in vertices:
        normalized_output.append(
            [
                float(vertex["x"] / original_width),
                float(vertex["y"] / original_height),
            ]
        )
        output.append([vertex["x"], vertex["y"]])
    polygon_is_valid = Polygon(
        vertices=[Point(x, y) for [x, y] in output]
    ).is_polygon_valid()
    if min_resolution:

        polygon = RectangleXYWH.from_polygon(
            Polygon(vertices=[Point(x, y) for [x, y] in output])
        )
        if polygon.w * polygon.h < min_resolution:
            logger.error(
                f"Resolution of polygon should be more than {min_resolution} "
                f"found {polygon.w * polygon.h}"
            )
            return None
    return normalized_output, polygon_is_valid


def get_scale_label(blob) -> dict:
    """Scale labels from s3 blob

    Args:
        blob (Any): blobs from s3 voxel-raw-labels

    Raises:
        ValueError: If the polygon is invalid

    Returns:
        dict: labels in dict
    """
    camera_uuid = blob.key.replace(SRC_PREFIX, "").replace(".json", "")
    result = {
        "doors": [],
        "driving_areas": [],
        "actionable_regions": [],
        "intersections": [],
        "end_of_aisles": [],
        "no_pedestrian_zones": [],
        "motion_detection_zones": [],
        "no_obstruction_regions": [],
    }

    task = json.loads(blob.get()["Body"].read())
    for annotation in task["response"]["annotations"]:
        annot = annotation.get("attributes", {})
        normalized_result = get_normalized_points_for_scale(
            annotation["vertices"],
            task["image_shape"]["height"],
            task["image_shape"]["width"],
            MIN_EXIT_DOOR_RESOLUTION
            if annotation["label"] == "door"
            and annot["type"].upper() == "EXIT"
            else MIN_DOOR_RESOLUTION
            if annotation["label"] == "door"
            else 0,
        )
        if normalized_result:
            annot["polygon"], polygon_is_valid = normalized_result
            result[f'{annotation["label"]}s'].append(annot)
            if not polygon_is_valid:
                # Raise error & fix (scale) here as this issue laters throws sentry error in PROD
                raise ValueError(
                    f"Invalid {annotation['label']} Polygon for camera {camera_uuid}"
                )
        else:
            logger.error(f"Unable to add {annotation} to the camera_config")
    door_types = {door["type"] for door in result["doors"]}
    result["doors"] = discard_different_door_type(door_types, result["doors"])
    return result, camera_uuid


def sync_camera_config(portal_environment, should_force_sync=False) -> None:
    """Sync camera config to portal database

    Args:
        portal_environment (str): portal environment
        should_force_sync (bool, optional): force sync. Defaults to False.

    Raises:
        ValueError: badly formatted xml
        RuntimeError: duplicated door ids in xml
    """
    camera_config_task_to_raw_labels(
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), 10, is_test=False
    )
    output = {}
    camera_uuids = set()
    is_valid = True
    for blob in get_blobs_from_bucket("voxel-raw-labels", "camera_config"):
        if ".json" in blob.key:
            result, camera_uuid = get_scale_label(blob)
            output[camera_uuid] = result
            camera_uuids.add(camera_uuid)

            if not validate_scale_and_graph_config_incidents(
                camera_uuid, result, True
            ):
                is_valid = False

    if not is_valid:
        if not should_force_sync:
            raise RuntimeError(
                (
                    "Please fix the above errors and re-run the script. "
                    "If you are sure that the above errors are not valid, "
                    "please pass the flag --force_sync"
                )
            )
    camera_config_response = update_portal_database(output, portal_environment)
    validate_camera_config_response(camera_config_response)


def update_portal_database(output: dict, portal_environment: str) -> dict:
    """Updates portal database with new camera configs

    Args:
        output (dict): camera configs from cvat
        portal_environment (str): environment to sync config to

    Returns:
        dict: response from portal calls

    Raises:
        RuntimeError: if camera config is not created
    """
    credentials = literal_eval(
        get_secret_from_aws_secret_manager(
            f"{portal_environment}_PERCEPTION_PORTAL_AUTH"
        )
    )
    camera_config_response = {}
    with requests.Session() as session:
        data = {
            "client_id": credentials["client_id"],
            "client_secret": credentials["client_secret"],
            "audience": credentials["audience"],
            "grant_type": "client_credentials",
        }
        response = session.post(credentials["auth_url"], data=data)
        access_token = json.loads(response.text)["access_token"]
        _host = credentials["host"]
        headers = {"authorization": f"Bearer {access_token}"}
        for uuid, config in output.items():
            if portal_environment == "STAGING":
                set_camera(uuid, headers, session, _host)
            query = f'mutation{{cameraConfigNewCreate(uuid:"{uuid}"'
            for key, value in config.items():
                query = f"{query},{underscore_to_camel(key)}:{json.dumps(json.dumps(value))}"
            query = (
                f"{query}) {{cameraConfigNew{{version,doors,drivingAreas,actionableRegions,"
                "intersections,endOfAisles,noPedestrianZones,motionDetectionZones, "
                "noObstructionRegions}, isUpdated}}"
            )
            logger.info(f"Query {query}")
            response = session.post(
                f"{_host}/graphql/", json={"query": query}, headers=headers
            )
            logger.info(
                f"Result of updating camera config {json.loads(response.text)}"
            )
            camera_config = json.loads(response.text)["data"][
                "cameraConfigNewCreate"
            ]
            if not camera_config:
                raise RuntimeError(
                    f"Unable to update camera config for camera uuid {uuid}"
                )
            camera_config_response[uuid] = camera_config

        # TODO: Automatically update and create a PR
        logger.info(
            "Update the camera config version in graph config.Also run "
            "bazel run core/execution/utils:fetch_camera_configs "
            "to update the camera_config.json and check in into github"
        )
        return camera_config_response


def parse_args() -> argparse.Namespace:
    """
    Parse input commandline args

    Returns:
        argparse.Namespace: the parsed input commanline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--portal_environment",
        type=str,
        default="STAGING",
        help="environment of portal",
    )
    parser.add_argument(
        "--force",
        type=str,
        default="false",
        help=("Force sync even with errors"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sync_camera_config(args.portal_environment, args.force == "true")
