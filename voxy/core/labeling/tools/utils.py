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
import os
import tempfile
from datetime import datetime

import botocore.exceptions
import cv2
import numpy as np
from loguru import logger

from core.labeling.tools.pull_kinesis_feed import get_frame_from_kinesis
from core.labeling.tools.queries import (
    STAGING_CAMERA_CREATE_QUERY,
    STAGING_ORGANIZATION_CREATE_QUERY,
    STAGING_ZONE_CREATE_QUERY,
)
from core.utils.aws_utils import upload_directory_to_s3


def draw_polygon(frame, polygon, color):
    """Draws polygon on frame

    Args:
        frame (np.ndarray): frame
        polygon (list): list of points
        color (tuple): color of polygon

    Returns:
        np.ndarray: frame with polygon drawn
    """
    height, width = frame.shape[:2]
    thickness = 2
    return cv2.polylines(
        frame,
        np.int32([np.array(polygon["polygon"]) * [width, height]]),
        True,
        color,
        thickness,
    )


def validate_camera_config_response(camera_config_response: dict) -> None:
    """Validates camera config response

    Note: Download AWS Toolkit to view the changed and unchanged files
    in AWS S3 bucket voxel-perception/sync_camera_config/<date>
    Args:
        camera_config_response (dict): response from portal calls

    Raises:
        RuntimeError: if camera config is not created
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        os.makedirs(os.path.join(temp_dir, "changed"))
        os.makedirs(os.path.join(temp_dir, "unchanged"))
        for camera_uuid, response in camera_config_response.items():
            try:
                frame = get_frame_from_kinesis(
                    camera_uuid, (datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
                )
            except botocore.exceptions.ClientError:
                logger.warning(
                    f"Could not retrieve frame from kinesis for {camera_uuid}"
                )
                continue
            attributes_color_map = {
                "doors": (0, 255, 0),
                "drivingAreas": (255, 0, 0),
                "actionableRegions": (0, 0, 255),
                "intersections": (160, 32, 240),
                "endOfAisles": (255, 165, 0),
                "noPedestrianZones": (165, 42, 42),
                "motionDetectionZones": (128, 128, 128),
                "noObstructionRegions": (196, 180, 84),
            }
            for attribute, color in attributes_color_map.items():
                for polygon in json.loads(
                    response["cameraConfigNew"][attribute]
                ):
                    draw_polygon(frame, polygon, color)
                    prefix = (
                        "changed" if response["isUpdated"] else "unchanged"
                    )
                    path = os.path.join(
                        temp_dir,
                        prefix,
                        f'{"_".join(camera_uuid.split("/"))}_{response["isUpdated"]}.jpg',
                    )
                    cv2.imwrite(
                        path,
                        frame,
                    )

        s3_path = upload_directory_to_s3(
            bucket="voxel-perception",
            local_directory=temp_dir,
            s3_directory=f"sync_camera_config/{datetime.now().strftime('%Y_%m_%d')}",
        )
        logger.info(
            "Download AWS Toolkit to view the changed and unchanged files",
            "in AWS S3 bucket voxel-perception/sync_camera_config/<date>",
        )
        logger.info(f"Camera config validation images uploaded to {s3_path}")


def set_camera(uuid, headers, session, host) -> None:
    """Create camera for staging environment

    Args:
        uuid (str): camera uuid
        headers (dict): post session headers
        session (Session): rest session
        host (str): name of the host
    """
    camera_array = uuid.split("/")
    variables = {
        "organizationName": camera_array[0],
        "organizationKey": camera_array[0],
    }
    session.post(
        f"{host}/graphql/",
        json={
            "query": STAGING_ORGANIZATION_CREATE_QUERY,
            "variables": variables,
        },
        headers=headers,
    )
    variables = {
        "zoneKey": camera_array[2],
        "zoneName": camera_array[2],
        "zoneType": "site",
        "organizationKey": camera_array[0],
    }
    session.post(
        f"{host}/graphql/",
        json={
            "query": STAGING_ZONE_CREATE_QUERY,
            "variables": variables,
        },
        headers=headers,
    )
    variables = {
        "cameraUuid": uuid,
        "cameraName": uuid,
        "zoneKey": camera_array[2],
        "organizationKey": camera_array[0],
    }
    session.post(
        f"{host}/graphql/",
        json={
            "query": STAGING_CAMERA_CREATE_QUERY,
            "variables": variables,
        },
        headers=headers,
    )
