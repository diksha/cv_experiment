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
from datetime import datetime

from loguru import logger

from core.labeling.tools.pull_kinesis_feed_site import get_feeds_for_site
from core.ml.data.curation.check_camera_uuid import date_after_retention_period

# trunk-ignore-all(pylint/R0801)


class PipelineIngestException(Exception):
    pass


def organization_has_cameras_online(location: str, organization: str) -> bool:
    """Whether organization has cameras that are online

    Args:
        location (str): location of cameras
        organization (str): organization cameras belong to

    Returns:
        bool: whether they are online
    """
    cameras_in_kinesis = get_feeds_for_site(
        organization=organization, location=location
    )
    for camera in cameras_in_kinesis:
        logger.info(f" Found camera: {camera}")
    if not cameras_in_kinesis:
        logger.error("❌ No cameras found online in kinesis")
        return False
    return True


def check_location_site(
    organization: str, location: str, ingestion_datetime: datetime
) -> bool:
    """
    Verifies the input arguments, date, time and the location and site

    Args:
        organization (str): the organization to verify
        location (str): the location to check
        ingestion_datetime (datetime): datetime to start ingestion

    Returns:
        bool: validating the cameras for location and site
    """
    if not organization_has_cameras_online(
        organization=organization, location=location
    ):
        logger.error(
            f"The organization and location: {organization} {location} has no cameras online"
        )
    logger.info(f"✅  Organization: {organization}-{location} looks good!")

    if not date_after_retention_period(ingestion_datetime):
        logger.error(
            f"Date: {ingestion_datetime} is before the retention period"
        )
        return False
    logger.info("👍  Everything looks good")
    return True


def get_args():
    parser = argparse.ArgumentParser(
        description="Checks organization and location for active feeds as first step of yolo data curation pipeline"
    )
    parser.add_argument(
        "--organization",
        type=str,
        required=True,
        help="The organization to check",
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="The location to check",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - dt.timedelta(hours=24)).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--offset-time",
        default=0,
        type=float,
        help="The time of day in military time (UTC)."
        " For example, 6:00AM would be 6.0. Defaults to 12 AM",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()
    if not args.start_date or args.start_date == "yesterday":
        one_day = dt.timedelta(hours=24)
        args.start_date = (datetime.now() - one_day).strftime("%Y-%m-%d")
    ingestion_datetime_main = datetime.strptime(
        args.start_date, "%Y-%m-%d"
    ) + dt.timedelta(hours=args.offset_time)
    if not check_location_site(
        args.organization, args.location, ingestion_datetime_main
    ):
        raise PipelineIngestException("Camera could not be validated")
