import json
import os
from ast import literal_eval

import requests
from loguru import logger

from core.execution.utils.graph_config_utils import (
    validate_scale_and_graph_config_incidents,
)
from core.incidents.utils import CameraConfig, CameraConfigError
from core.utils.aws_utils import get_secret_from_aws_secret_manager
from core.utils.yaml_jinja import load_yaml_with_jinja

CAMERA_CONFIG_PATH = "configs/cameras/camera_config.json"
CAMERAS = "configs/cameras/cameras"
environment = "PROD"


def _get_camera_uuid_version(cameras):
    camera_uuid_version_map = {}
    for config_path in cameras:
        config = load_yaml_with_jinja(config_path)
        camera_uuid_version_map[config["camera_uuid"]] = config["camera"][
            "version"
        ]
    return camera_uuid_version_map


def _fetch_camera_configs(cameras):
    credentials = literal_eval(
        get_secret_from_aws_secret_manager(
            f"{environment}_PERCEPTION_PORTAL_AUTH"
        )
    )
    camera_uuid_version_map = _get_camera_uuid_version(cameras)
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
        camera_config = {}
        for uuid, version in camera_uuid_version_map.items():
            query = f'{{cameraConfigNew(uuid: "{uuid}", version: {version}) \
            {{doors, drivingAreas, actionableRegions, intersections, endOfAisles , \
            noPedestrianZones, motionDetectionZones, noObstructionRegions}}}}'
            response = session.post(
                f"{_host}/graphql/",
                json={"query": query},
                headers=headers,
            )
            mapping = {}

            logger.info(f"Running for camera: {uuid} version: {version}")

            for key, value in json.loads(response.text)["data"][
                "cameraConfigNew"
            ].items():
                if value:
                    mapping[key] = json.loads(value)
            validate_scale_and_graph_config_incidents(uuid, mapping, False)
            mapping["version"] = version
            camera_config[uuid] = mapping
            try:
                camera_conf = CameraConfig(uuid, 1, 1)
                next_door_id = camera_conf.next_door_id
            except CameraConfigError:
                next_door_id = 1
            # Set next door id
            max_door_id = (
                max(door["door_id"] for door in mapping["doors"])
                if mapping["doors"]
                else 0
            )
            mapping["nextDoorId"] = (
                max_door_id + 1
                if next_door_id <= max_door_id
                else next_door_id
            )

        return camera_config


def _write_camera_config(camera_config):
    camera_config_path = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"], CAMERA_CONFIG_PATH
    )

    with open(camera_config_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(camera_config, indent=4, sort_keys=True))


def fetch_camera_config_lib():
    with open(CAMERAS, encoding="utf-8") as f:
        cameras = []
        for line in f:
            cameras.append(line.rstrip("\n"))
        camera_config = _fetch_camera_configs(cameras)
        _write_camera_config(camera_config)
