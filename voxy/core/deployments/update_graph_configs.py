#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import argparse
import glob
import os
from typing import List, Optional, Union

import mergedeep
import yaml
from loguru import logger

from core.utils.logging.list_indented_yaml_dumper import ListIndentedDumper
from core.utils.yaml_jinja import load_yaml_with_jinja

GRAPH_CONFIG_PATH = "configs/cameras"
ALL_CAMERAS_WILDCARD = "**/**/**/*.yaml"
CONFIG_EXT = "yaml"
COPYRIGHT = (
    "#\n"
    "# Copyright 2020-2021 Voxel Labs, Inc.\n"
    "# All rights reserved.\n"
    "#\n"
    "# This document may not be reproduced, republished, distributed, transmitted,\n"
    "# displayed, broadcast or otherwise exploited in any manner without the express\n"
    "# prior written permission of Voxel Labs, Inc. The receipt or possession of this\n"
    "# document does not convey any rights to reproduce, disclose, or distribute its\n"
    "# contents, or to manufacture, use, or sell anything that it may describe, in\n"
    "# whole or in part.\n"
    "#\n"
    "#\n"
)


def parse_args() -> argparse.Namespace:
    """Parse arguments
    Returns:
        argparse.Namespace: argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        "-config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--camera-uuids",
        "-cameras",
        nargs="+",
        type=str,
        default=[],
        required=False,
    )
    parser.add_argument(
        "--organization",
        "-org",
        required=False,
    )
    parser.add_argument(
        "--location",
        "-loc",
        required=False,
    )
    parser.add_argument(
        "--update-all-cameras",
        "-all",
        action="store_true",
    )
    return parser.parse_args()


def generate_cameras_wildcard(
    organization: Optional[str],
    location: Optional[str],
    all_cameras: bool,
) -> Optional[str]:
    """Generate wildcard if user specified via inputs
    Args:
        organization (Optional[str]): organization
        location (Optional[str]): location
        all_cameras (bool): flag specifying to use all cameras
    Returns:
        Optional[str]: wildcard
    """
    if all_cameras:
        return ALL_CAMERAS_WILDCARD
    if organization or location:
        wildcard_split = ALL_CAMERAS_WILDCARD.split("/")
        wildcard_split[0] = organization if organization is not None else "**"
        wildcard_split[1] = location if location is not None else "**"
        return ("/").join(wildcard_split)
    return None


def get_cameras(wildcard: str) -> List[str]:
    """Get all camera uuids present in configs/cameras
    Args:
        wildcard (str): wildcard to search for configs
    Returns:
        List[str]: all camera uuids
    """
    config_paths = glob.glob(os.path.join(GRAPH_CONFIG_PATH, wildcard))
    camera_uuids = [
        os.path.splitext(c.split(f"{GRAPH_CONFIG_PATH}/")[1])[0]
        for c in config_paths
    ]
    return camera_uuids


def remove_items(
    dict_to_keep: dict, items_to_remove: Union[dict, list, str]
) -> dict:
    """Remove dict_to_remove keys from dict_to_keep
    Args:
        dict_to_keep (dict): dictionary to have items removed
        items_to_remove (Union[dict, list, str]): items to remove, either list of
            keys, or subdictionary
    Returns:
        dict: dict_to_keep with items removed
    Raises:
        RuntimeError: ivalid datatype in items_to_remove
    """
    if isinstance(items_to_remove, str):
        if items_to_remove in dict_to_keep:
            del dict_to_keep[items_to_remove]
        else:
            logger.warning(
                f"Key to remove does not exist at current layer, {items_to_remove}"
            )
        return dict_to_keep

    if isinstance(items_to_remove, list):
        for item in items_to_remove:
            remove_items(dict_to_keep, item)

    if isinstance(items_to_remove, dict):
        for key, value in items_to_remove.items():
            if key not in dict_to_keep:
                logger.warning(
                    f"Node key does not exist at current layer, {key}, noop"
                )
                remove_items(dict_to_keep, [])
            else:
                remove_items(dict_to_keep[key], value)

    return dict_to_keep


def update_camera_configs(config_path: str, camera_uuids: List[str]) -> None:
    """Update camera configs with config items
    Args:
        config_path (str): path to config containing things to add
        camera_uuids (List[str]): list of camera_uuids to update
    """
    logger.info(
        f"Updating configs for the following camera_uuids, {camera_uuids}"
    )
    config = load_yaml_with_jinja(config_path)
    items_to_ammend = config.get("items_to_amend", {})
    items_to_remove = config.get("items_to_remove", [])
    for camera_uuid in camera_uuids:
        filename = f"{camera_uuid}.{CONFIG_EXT}"
        graph_config = load_yaml_with_jinja(
            os.path.join(GRAPH_CONFIG_PATH, filename)
        )
        mergedeep.merge(graph_config, items_to_ammend)
        remove_items(graph_config, items_to_remove)
        with open(
            os.path.join(GRAPH_CONFIG_PATH, filename), "w", encoding="UTF-8"
        ) as config:
            config.write(COPYRIGHT)
            yaml.dump(graph_config, config, Dumper=ListIndentedDumper)


if __name__ == "__main__":
    args = parse_args()
    cameras = args.camera_uuids
    camera_wildcard = generate_cameras_wildcard(
        args.organization,
        args.location,
        args.update_all_cameras,
    )
    if camera_wildcard is not None:
        cameras.extend(get_cameras(camera_wildcard))
    cameras = list(set(cameras))
    update_camera_configs(
        args.config_path,
        cameras,
    )
