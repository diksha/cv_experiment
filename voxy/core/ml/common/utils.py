import argparse
import typing
from typing import List, Optional

import mergedeep
from loguru import logger

from core.utils.camera_config_utils import (
    get_camera_uuids_from_organization_site,
)


def get_merged_config(config, overwrite_config=None) -> dict:
    """Given config and overwrite get merged config

    Args:
        config (dict): original config
        overwrite_config (dict): overwrite config

    Returns:
        dict: merged config
    """
    if not overwrite_config:
        return config
    return mergedeep.merge(config, overwrite_config)


# TODO: we want to be able to create tasks using # trunk-ignore(pylint/W0511)
# a separate pipeline and workflow and hopefully this becomes obselete
def add_camera_uuid_parser_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Adds extra arguments for pulling camera uuids required for task creation. This should be used
    for a common code path for all task creation/dummy task creation objects

    Args:
        parser (argparse.ArgumentParser): the parser

    Returns:
        argparse.ArgumentParser: the modified parser with the extra camera uuid arguments
    """
    parser.add_argument(
        "--camera_uuids",
        type=str,
        default=[],
        nargs="*",
        help=(
            "The camera uuids to generate the task. The priority is defined as: "
            "(1) Camera uuids, (2) organization and location, (3) organization (4) error"
        ),
    )
    parser.add_argument(
        "--organization",
        type=str,
        help=(
            "The organization to associate with this task. The priority is defined as: "
            "(1) Camera uuids, (2) organization and location, (3) organization (4) error"
        ),
        required=False,
        default=None,
    )
    parser.add_argument(
        "--location",
        type=str,
        help=(
            "The location to associate with this task"
            "The priority is defined as: "
            "(1) Camera uuids, (2) organization and location, (3) organization (4) error"
        ),
        required=False,
        default=None,
    )
    return parser


def get_camera_uuids_from_arguments(
    camera_uuids: List[str],
    organization: Optional[str],
    location: Optional[str],
) -> typing.List[str]:
    """
    Generates the list of cameras of camera uuids from the arguments

    Args:
        camera_uuids (List[str]): list of cameras
        organization (Optional[str]): orgaization
        location (Optional[str]): location

    Raises:
        ValueError: when there was no valid camera uuid or organization found.
                The organization/location field or the camera uuid field must be
                populated

    Returns:
        typing.List[str]: the list of camera uuids as determined
                          from the input commandline arguments
    """

    if camera_uuids:
        return camera_uuids
    if organization:
        return get_camera_uuids_from_organization_site(organization, location)
    logger.info("No camera arguments passed, assuming all cameras")
    return []
