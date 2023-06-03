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
from typing import List

import boto3
import sematic
from loguru import logger
from sematic.resolvers.silent_resolver import SilentResolver

from core.labeling.tools.pull_kinesis_feed import PullFeedResult, pull_feed


def stream_name_to_uuid(stream_name: str) -> str:
    """
    Converts the stream name (organization-location-zone) to a camera uuid format
    (organization/location/zone/cha)

    Args:
        stream_name (str): raw stream name (organization-location-zone)

    Returns:
        str: the interpretted uuid name
    """
    uuid = stream_name.replace("-", "/") + "/cha"
    # the camera uuid is usually something like:
    # americold/modesto/0001/cha
    # but sometimes it's
    # americold/savannah_bloomingdale/0001/cha
    # the candidate would be
    # americold/savannah/bloomingdale/0001/cha
    # we want to replace this intermediate /
    if uuid.count("/") > 3:
        extra_slash = uuid.count("/") - 3
        # we seek to the second / and keep removing until we have no extra slashes
        while extra_slash > 0:
            location = uuid.find("/", uuid.find("/") + 1)
            list_uuid = list(uuid)
            list_uuid[location] = "_"
            uuid = "".join(list_uuid)
            extra_slash -= 1

    return uuid


def get_feeds_for_site(organization: str, location: str) -> list:
    """
    Grabs the list of feeds for a particular site and location that are available in kinesis

    Args:
        organization (str): the organization (e.g. americold)
        location (str): the current location (e.g. modesto)

    Returns:
        list: list of all the valid stream uuids
    """
    AWS_REGION = "us-west-2"
    MAX_STREAMS_PER_LOCATION = 100
    client = boto3.Session(profile_name="production").client(
        "kinesisvideo", region_name=AWS_REGION
    )
    query = f"{organization}-{location}"

    response = client.list_streams(
        MaxResults=MAX_STREAMS_PER_LOCATION,
        StreamNameCondition={
            "ComparisonOperator": "BEGINS_WITH",
            "ComparisonValue": query,
        },
    )
    organization_streams = [
        stream["StreamName"]
        for stream in response["StreamInfoList"]
        if stream["Status"] == "ACTIVE"
    ]
    return [stream_name_to_uuid(stream) for stream in organization_streams]


@sematic.func
# trunk-ignore(pylint/R0913)
def pull_kinesis_feed_site(
    ingestion_datetime: datetime,
    hours: float,
    organization: str,
    location: str,
    max_fragment_failures: int,
    bucket: str,
    prefix: str,
    specific_camera_uuids: object = None,
    chunk_size: int = 10,
    max_videos: int = 0,
) -> List[PullFeedResult]:
    """
    Pulls kinesis feed for a specific organization/location

    Args:
        ingestion_datetime (str): the start date to begin pulling from
        hours (float): the number of hours to pull the video
        organization (str): the organization to pull the feed for (i.e. americold)
        location (str): the location to pull the feed (i.e. modesto)
        max_fragment_failures (int): the maximum number of fragment failures
        bucket (str): the output buckets to put the clips
        prefix (str): the prefix for the output bucket to put the videos
        specific_camera_uuids (object): Specific_camera_uuids list, or None if not specified
        chunk_size (int, optional): The chunk size in minutes. Defaults to 10.
        max_videos (int, optional): Maximum number of videos per camera.  Default 0 is unlimited

    Returns:
        List[PullFeedResult]: List of camera feeds and s3 location of videos
    """
    # query for the available streams
    cameras_in_kinesis = get_feeds_for_site(
        organization=organization, location=location
    )
    cameras_in_kinesis = (
        filter(
            lambda camera: camera in specific_camera_uuids, cameras_in_kinesis
        )
        if specific_camera_uuids is not None
        else cameras_in_kinesis
    )

    pull_kinesis_feed_results = []
    for camera_uuid in cameras_in_kinesis:
        logger.info(f"Pulling feed for {camera_uuid}")
        pull_kinesis_feed_results.append(
            pull_feed(
                ingestion_datetime,
                hours,
                chunk_size,
                camera_uuid,
                bucket,
                max_fragment_failures,
                prefix,
                max_videos=max_videos,
            )
        )

    logger.info("Pulled kinesis feed future")
    return pull_kinesis_feed_results


def parse_args():
    parser = argparse.ArgumentParser()
    # object_path is relative to --bucket
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - dt.timedelta(hours=24)).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help="The location to pull feeds from, e.g. Modesto",
    )
    parser.add_argument(
        "--organization",
        type=str,
        required=True,
        help="The organization to pull feeds from, e.g. Americold",
    )
    parser.add_argument(
        "--hours",
        default=24,
        type=float,
        help="The number of hours for the kinesis stream pull",
    )
    parser.add_argument(
        "--offset-time",
        default=0,
        type=float,
        help="The time of day in military time (UTC). For example, 6:00AM would be 6.0. Defaults to 12 AM",
    )
    parser.add_argument(
        "--chunk_size", default=10, type=int, help="Chunk size in minutes"
    )
    parser.add_argument(
        "--bucket",
        required=False,
        default="voxel-lightly-input",
        type=str,
        help="The bucket to store the kinesis streams",
    )
    parser.add_argument(
        "--max-fragment-failures",
        required=False,
        type=int,
        default=5,
        help="Maximum number of fragment failures before throwing an error",
    )
    parser.add_argument(
        "--prefix",
        required=False,
        default="",
        type=str,
        help="The prefix to add to the bucket output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.start_date or arguments.start_date == "yesterday":
        one_day = dt.timedelta(hours=24)
        arguments.start_date = (datetime.now() - one_day).strftime("%Y-%m-%d")

    ingestion_datetime_main = start_time = datetime.strptime(
        arguments.start_date, "%Y-%m-%d"
    ) + dt.timedelta(hours=arguments.offset_time)
    pull_kinesis_feed_results_main = pull_kinesis_feed_site(
        ingestion_datetime=ingestion_datetime_main,
        hours=arguments.hours,
        organization=arguments.organization,
        location=arguments.location,
        max_fragment_failures=arguments.max_fragment_failures,
        bucket=arguments.bucket,
        prefix=arguments.prefix,
        chunk_size=arguments.chunk_size,
    ).resolve(SilentResolver())
    FAILED_CAMERAS = 0
    for key, value in pull_kinesis_feed_results_main:
        if not value:
            logger.error(f"Failed for {key}")
            FAILED_CAMERAS += 1
    if FAILED_CAMERAS == len(pull_kinesis_feed_results_main):
        raise RuntimeError("All kinesis streams are down")
    logger.info(f"Number of failed cameras is {FAILED_CAMERAS}")
