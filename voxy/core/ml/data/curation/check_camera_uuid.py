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

import argparse
import datetime as dt
import json
import os
from datetime import datetime

from loguru import logger

RETENTION_PERIOD_DAYS = 7


def attribute_present(attribute: str, config: dict) -> bool:
    if attribute not in config:
        logger.error(f"❌ Attribute: '{attribute}' was not found")
        return False
    return True


def check_door_config(door_config: dict) -> bool:
    """
    Checks to see if there are the required information like orientation,
       door id, polygon and type


    Args:
        door_config (dict): the dictionary from the camera config

    Returns:
        bool: if the door config is valid
    """
    attributes = ["orientation", "door_id", "polygon", "type"]
    for attribute in attributes:
        if not attribute_present(attribute, door_config):
            return False
    return True


def camera_config_valid(camera_uuid: str) -> bool:
    """
    Checks if the camera config valid. These are some basic checks implemented:
    1. checks to see if the top level camera config exists
    2. checks to see if the uuid exists in the camera config
    3. checks to see if doors are present in the config
    4. checks to see if there are the required information like orientation,
       door id, polygon and type

    Args:
        camera_uuid (str): the camera uuid to check

    Returns:
        bool: whether or not the camera config is valid
    """
    camera_config_path = "configs/cameras/camera_config.json"
    if not os.path.exists(camera_config_path):
        logger.error(f"❌ Camera config: {camera_config_path} doesn't exist")
        return False
    logger.info("📄 Found top level camera config")
    with open(
        camera_config_path,
        "r",
        encoding="utf8",
    ) as camera_config_file:
        camera_config = json.load(camera_config_file)
        config_for_uuid = camera_config.get(camera_uuid)
        if config_for_uuid is None:
            logger.error(
                f"❌ Camera config for camera uuid: {camera_uuid} in {camera_config_path} doesn't exist"
            )
            return False
        logger.info("📄 Found config for camera uuid")
        doors = config_for_uuid.get("doors")
        if doors is None or len(doors) < 1:
            logger.error(
                f"❌ Camera config for camera uuid: {camera_uuid} in {camera_config_path} does not have any doors"
            )
            return False
        logger.info("🚪 Found doors in config")
        for idx, door in enumerate(doors):
            if check_door_config(door):
                logger.info(
                    f"🚪 Door @ index {idx} (door_id) in config was valid"
                )
            else:
                logger.error(
                    f"❌ Door @ index {idx} in {camera_config_path} was invalid: {door}"
                )
                return False
    return True


def date_after_retention_period(date: dt.date) -> bool:
    """
    Checks to see if the date is before the retention period. The retention
    period is assumed to be RETENTION_PERIOD_DAYS days

    Args:
        date (dt.date): the date to compare

    Returns:
        bool: whether the date and time is longer than RETENTION_PERIOD_DAYS ago
    """
    if date < datetime.now() - dt.timedelta(days=RETENTION_PERIOD_DAYS):
        logger.error("❌ Date was before the retention period")
        return False
    return True


class PipelineIngestException(Exception):
    pass


def main(args):
    if not camera_config_valid(args.camera_uuid):
        raise PipelineIngestException(
            f"The camera config for camera: {args.camera_uuid} is invalid"
        )
    logger.info(f"✅  Camera config for {args.camera_uuid} looks good")
    logger.info(f"👍  Camera uuid:{args.camera_uuid} looks good")

    if not args.start_date:
        raise PipelineIngestException("Start date required!")

    try:
        start_time = datetime.strptime(
            args.start_date, "%Y-%m-%d"
        ) + dt.timedelta(hours=args.offset_time)
    except Exception:
        logger.exc("Invalid start date")
        # trunk-ignore(pylint/W0707)
        raise PipelineIngestException(f"Invalid start date {args.start_date}")

    if not args.skip_retention_check and not date_after_retention_period(
        start_time
    ):
        raise PipelineIngestException(
            f"Date: {args.start_date} @ {args.offset_time} hours is before the retention period"
        )
    logger.info(
        f"👍  Date: {args.start_date} and time {args.offset_time} hours looks good"
    )


def get_args():
    parser = argparse.ArgumentParser(description="Check camera uuid")
    parser.add_argument(
        "--camera-uuid",
        type=str,
        required=True,
        help="The camera uuid",
    )
    parser.add_argument(
        "--start-date",
        type=str,
    )
    parser.add_argument(
        "--offset-time",
        default=0,
        type=float,
        help="The time of day in military time (UTC)."
        " For example, 6:00AM would be 6.0. Defaults to 12 AM",
    )
    parser.add_argument(
        "--skip-retention-check",
        action="store_true",
        help="Skip video retention checks (for data flywheel)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arguments = get_args()
    if not arguments.start_date or arguments.start_date == "yesterday":
        one_day = dt.timedelta(hours=24)
        arguments.start_date = (datetime.now() - one_day).strftime("%Y-%m-%d")
    main(arguments)
