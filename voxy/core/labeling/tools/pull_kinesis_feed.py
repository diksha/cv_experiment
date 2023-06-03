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
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import av
import boto3
import botocore
import numpy
import sematic
from loguru import logger
from sematic.resolvers.silent_resolver import SilentResolver
from tqdm import tqdm

from core.infra.sematic.shared.resources import CPU_2CORE_8GB
from core.utils.aws_utils import upload_file
from core.utils.yaml_jinja import load_yaml_with_jinja

# Pulls the kinesis stream video from the target date
#
# usage: pull_kinesis_feed.py [-h] [--start_date START_DATE] [--hours HOURS]
# [--chunk_size CHUNK_SIZE] --camera-uuid CAMERA_UUID
# [--output-bucket OUTPUT_BUCKET]
#
# optional arguments:
#   -h, --help            show this help message and exit
#   --start_date START_DATE
#   --hours HOURS         The number of hours for the kinesis stream pull
#   --chunk_size CHUNK_SIZE
#                         Chunk size in minutes
#   --camera-uuid CAMERA_UUID
#                         The camera uuid to pull the stream from
#   --output-bucket OUTPUT_BUCKET
#                         The output bucket to store the kinesis streams
#
# For example:
#       for a dev machine:
#
# $ ./bazel run //core/labeling/tools:pull_kinesis_feed -- --camera-uuid americold/ontario/0005/cha
#
#       for a cloud agent w/ credentials:
#
# $ ./bazel run //core/labeling/tools:pull_kinesis_feed -- --camera-uuid americold/ontario/0005/cha


AWS_REGION = "us-west-2"


@dataclass
class PullFeedResult:
    """A summary of pull kinesis feed
    Attributes
    ----------
    camera_uuid:
        The UUID of the camera whose video was ingested
    s3_path:
        Path where videos are ingested
    """

    camera_uuid: str
    s3_path: Optional[str]


def get_clips(
    start_time: datetime.time,
    end_time: datetime.time,
    chunk_size_minutes: int = 10,
):
    """
    Iterable generator that creates all the sequences from the start to the end timestamp

    Args:
        start_time (datetime.time): the time to begin
        end_time (datetime.time): the ending time
        chunk_size_minutes (int): The chunk size for full time range. Defaults to 10 minutes.

    Yields:
        tuple: (start timestamp ms, end timestamp ms)
    """

    chunk_time_minutes = dt.timedelta(minutes=chunk_size_minutes)
    current_start_time = start_time
    current_end_time = start_time + chunk_time_minutes
    while current_start_time < end_time:
        current_start_ms = current_start_time.timestamp() * 1000
        current_end_ms = current_end_time.timestamp() * 1000
        yield (current_start_ms, current_end_ms)
        current_start_time += chunk_time_minutes
        current_end_time += chunk_time_minutes


def download_from_kinesis(
    stream_name: str, start_timestamp: int, end_timestamp: int
):
    """
    Downloads a clip from kinesis using the start and end timestamp.

    Args:
        stream_name (str): the kinesis video stream name, usually in the format: 'organization-site-camera-number'
        start_timestamp (int): the start timestamp (epoch seconds)
        end_timestamp (int): the end timestamp (epoch seconds)

    Returns:
        amazon structured payload for the video fragments
    """
    kinesis_client = boto3.Session(profile_name="production").client(
        "kinesisvideo", region_name=AWS_REGION
    )
    kinesis_endpoint = kinesis_client.get_data_endpoint(
        StreamName=stream_name, APIName="GET_CLIP"
    )
    endpoint_url = kinesis_endpoint["DataEndpoint"]

    client = boto3.Session(profile_name="production").client(
        "kinesis-video-archived-media",
        endpoint_url=endpoint_url,
        region_name=AWS_REGION,
    )
    clip_fragment = {
        "FragmentSelectorType": "PRODUCER_TIMESTAMP",
        "TimestampRange": {
            "StartTimestamp": start_timestamp,
            "EndTimestamp": end_timestamp,
        },
    }

    try:
        response = client.get_clip(
            StreamName=stream_name,
            ClipFragmentSelector=clip_fragment,
        )
        return response["Payload"]

    except client.exceptions.ResourceNotFoundException as e:
        logger.error(
            f"Error downloading clip from kinesis. No fragments found in clip, skipping.\n {e}"
        )
        return None


def get_stream_name_from_camera_uuid(camera_uuid: str) -> str:
    """
    Generates the stream name from the camera uuid

    Args:
        camera_uuid (str): the camera uuid to generate the stream name. 'organization/site/channel/cha'

    Raises:
        ValueError: if the camera uuid is invalid

    Returns:
        str: the interpretted stream name from the camera uuid.
    """
    # try to find the camera configs
    if len(camera_uuid.split("/")) != 4:
        raise ValueError(f"The camera uuid: {camera_uuid} is invalid")
    stream_name = "-".join(camera_uuid.split("/")[:-1])
    return stream_name


def write_payload_to_file(
    payload: botocore.response.StreamingBody, filename: str
) -> str:
    """
    Writes boto streaming payload to the file

    Args:
        payload (botocore.reponse.StreamingBody): the botocore streaming payload
        filename (str): the file to write to

    Returns:
        str: the output filename
    """
    with open(filename, "wb") as out_file:
        for chunk in payload.iter_chunks():
            out_file.write(chunk)
    return filename


class KinesisStreamFragmentException(RuntimeError):
    pass


def update_fragment_failures(
    current_n_failures: int, max_n_failures: int
) -> int:
    """
    Update number of fragment failures

    Args:
        current_n_failures (int): the current number of failures
        max_n_failures (int): the maximum allowable number of failures

    Raises:
        RuntimeError: raised if the current failure count is exceeded

    Returns:
        int: the updated number of failures
    """
    updated_n_failures = current_n_failures + 1
    if updated_n_failures > max_n_failures:
        raise KinesisStreamFragmentException(
            "Maximum number of fragment failures exceeded! "
        )
    return updated_n_failures


def get_frame_from_kinesis(camera_uuid: str, frame_time: str) -> numpy.ndarray:
    """Get frame from kinesis given camera uuid

    Args:
        camera_uuid (str): uuid of the camera
        frame_time (str): When to get frame from

    Raises:
        RuntimeError: Unable to get frame

    Returns:
        numpy.ndarray: frame as nd array
    """
    config = load_yaml_with_jinja(f"configs/cameras/{camera_uuid}.yaml")
    arn = config["camera"]["arn"]
    start_time = (
        datetime.strptime(frame_time, "%Y-%m-%d %H:%M:%S")
        - dt.timedelta(minutes=10)
    ).timestamp()
    end_time = datetime.strptime(frame_time, "%Y-%m-%d %H:%M:%S").timestamp()
    kinesis_client = boto3.Session(profile_name="production").client(
        "kinesisvideo", region_name=AWS_REGION
    )
    kinesis_endpoint = kinesis_client.get_data_endpoint(
        StreamARN=arn, APIName="GET_CLIP"
    )
    endpoint_url = kinesis_endpoint["DataEndpoint"]
    client = boto3.Session(profile_name="production").client(
        "kinesis-video-archived-media",
        endpoint_url=endpoint_url,
        region_name=AWS_REGION,
    )
    clip_fragment = {
        "FragmentSelectorType": "PRODUCER_TIMESTAMP",
        "TimestampRange": {
            "StartTimestamp": start_time,
            "EndTimestamp": end_time,
        },
    }
    response = client.get_clip(
        StreamARN=arn,
        ClipFragmentSelector=clip_fragment,
    )
    payload = response["Payload"]
    with tempfile.NamedTemporaryFile() as temp:
        filename = temp.name
        with open(filename, "wb") as out_file:
            for chunk in payload.iter_chunks():
                out_file.write(chunk)
            out_file.flush()
        _video_container = av.open(filename)
        _video_stream = _video_container.streams.video[0]
        for frame in _video_container.decode(_video_stream):
            return av.VideoFrame.to_ndarray(frame, format="bgr24")

    raise RuntimeError("Unable to get frame from kinesis")


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func(
    resource_requirements=CPU_2CORE_8GB,
    standalone=True,
    retry=sematic.RetrySettings(exceptions=(Exception,), retries=4),
)
# trunk-ignore(pylint/R0913)
def pull_feed(
    ingestion_datetime: datetime,
    hours: float,
    chunk_size: int,
    camera_uuid: str,
    output_bucket: str,
    max_fragment_failures: int,
    prefix: str,
    postfix: str = "",
    max_videos: int = 0,
) -> PullFeedResult:
    """# Download a video stream from Kinesis and upload it to S3

    ## Parameters
    - **start_date**:
        The start date of the video stream to be pulled
    - **offset_time**:
        The number of hours into the day to pull (1.0 is 1am)
    - **hours**:
        The number of hours worth of video to pull
    - **chunk_size**:
        The number of minutes each video chunk should be
    - **camera_uuid**:
        The UUID of the camera whose feed should be pulled
    - **output_bucket**:
        The s3 bucket where the results should be pushed
    - **max_fragment_failures**:
        The max allowed number of fragment failures before an error will be raised
    - **max_videos**:
        Maximum number of videos to pull.  0 is unlimited.

    ## Returns
    Camera uuid and s3 location if kinesis pull was successful
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    duration = dt.timedelta(hours=hours)
    chunk_size_duration = dt.timedelta(minutes=chunk_size)
    end_time = ingestion_datetime + duration
    stream_name = get_stream_name_from_camera_uuid(camera_uuid)
    n_fragment_failures = 0
    for n_videos, (start_ms, end_ms) in tqdm(
        enumerate(
            get_clips(
                ingestion_datetime, end_time, chunk_size_minutes=chunk_size
            )
        ),
        total=int(duration / chunk_size_duration),
        ncols=100,
    ):
        if max_videos and n_videos >= max_videos:
            break
        output_file = (
            f'{ingestion_datetime.strftime("%Y-%m-%d")}'
            f"_{int(start_ms)}_ms_{int(end_ms)}_ms.mp4"
        )
        s3_filename = os.path.join(prefix, camera_uuid, postfix, output_file)

        logger.info(f"uploading to {s3_filename}")

        with tempfile.NamedTemporaryFile() as temporary_file:

            payload = download_from_kinesis(
                stream_name, int(start_ms // 1000), int(end_ms // 1000)
            )

            if payload is not None:
                write_payload_to_file(payload, temporary_file.name)
                upload_file(output_bucket, temporary_file.name, s3_filename)
            else:
                try:
                    n_fragment_failures = update_fragment_failures(
                        n_fragment_failures, max_fragment_failures
                    )
                except KinesisStreamFragmentException:
                    logger.warning(f"Unable to pull feed for {camera_uuid}")
                    return PullFeedResult(camera_uuid, None)
    return PullFeedResult(
        camera_uuid,
        f"s3://{os.path.join(output_bucket, prefix, camera_uuid, postfix)}",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    # object_path is relative to --bucket
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - dt.timedelta(hours=24)).strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--hours",
        default=1,
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
        "--camera-uuid",
        required=True,
        type=str,
        help="The camera uuid to pull the stream from",
    )
    parser.add_argument(
        "--output-bucket",
        required=False,
        default="voxel-lightly-input",
        type=str,
        help="The output bucket to store the kinesis streams",
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
        help="The prefix to move the videos to. The output will be s3://ouput-bucket/prefix/...",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.start_date or arguments.start_date == "yesterday":
        one_day = dt.timedelta(hours=24)
        arguments.start_date = (datetime.now() - one_day).strftime("%Y-%m-%d")
    ingestion_datetime_main = datetime.strptime(
        arguments.start_date, "%Y-%m-%d"
    ) + dt.timedelta(hours=arguments.offset_time)
    pull_feed(
        ingestion_datetime_main,
        arguments.hours,
        arguments.chunk_size,
        arguments.camera_uuid,
        arguments.output_bucket,
        arguments.max_fragment_failures,
        arguments.prefix,
    ).resolve(SilentResolver())
