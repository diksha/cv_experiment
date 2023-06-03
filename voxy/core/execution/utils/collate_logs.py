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
import datetime
import tempfile
import time
import typing
import uuid
from dataclasses import dataclass

from loguru import logger
from mcap_protobuf.reader import read_protobuf_messages
from mcap_protobuf.writer import Writer

from core.utils.aws_utils import (
    download_to_file,
    separate_bucket_from_relative_path,
    upload_file,
)

# Example usage:
# ./bazel run //core/execution/utils:collate_logs -- --s3_paths
#                   s3://voxel-temp/logs/path/to/log.mcap
#                   s3://voxel-temp/logs/path/to/other_log.mcap


@dataclass
class AggregatedResult:
    names: typing.List[str]
    s3_paths: typing.List[str]
    common_path: str


@dataclass
class MCAPMessage:
    """
    Basic wrapper for protobuf messages
    """

    proto: typing.Any
    topic: str
    timestamp: int


def download_file_from_s3(s3_filename: str, local_file: str):
    """
    Helper function to download a file from s3 given an s3 path

    Args:
        s3_filename (str): the s3 filename
        local_file (str): the local file path to download to
    """
    bucket, s3_path = separate_bucket_from_relative_path(s3_filename)
    download_to_file(bucket=bucket, local_path=local_file, s3_path=s3_path)


def convert_datetime_to_nanoseconds(date: datetime.datetime) -> int:
    """
    Converts datetime object to an integer in nanoseconds

    Args:
        date (datetime.datetime): the original date

    Returns:
        int: the date time in nanoseconds
    """
    return int(time.mktime(date.timetuple()) * 1e9)


def mcap_message_generator(
    s3_mcap_logs: typing.List[str], names: typing.List[str]
) -> MCAPMessage:
    """
    Iterator for seeking through mcap logs and
    returning all messages in the logs

    Args:
        s3_mcap_logs (typing.List[str]): the list of
             mcap logs to process
        names (typing.List[str]): the names of the logs (parent topic)

    Yields:
        Iterator[MCAPMessage]: the stream of mcap messages
    """
    for name, (i, mcap_log) in zip(names, enumerate(s3_mcap_logs)):
        logger.info(f"Processing (Run {i}): {mcap_log}")
        with tempfile.NamedTemporaryFile() as tmp:
            download_file_from_s3(mcap_log, tmp.name)
            for message in read_protobuf_messages(
                source=tmp.name, log_time_order=True
            ):
                yield MCAPMessage(
                    topic=f"/{name}/{message.topic.lstrip('/')}",
                    proto=message.proto_msg,
                    timestamp=convert_datetime_to_nanoseconds(
                        message.log_time
                    ),
                )


def aggregated_incidents_from_args(args: argparse.Namespace):
    """
    Gets aggregated incidents from commandline args

    Args:
        args (argparse.Namespace): commandline args

    Returns:
        typing.List[AggregatedResult]: the aggregated results
    """
    return AggregatedResult(
        names=[f"Run_{i}" for i in range(len(args.s3_logs))],
        s3_paths=args.s3_logs,
        common_path=f"{uuid.uuid4()}",
    )


def collate_logs(aggregated_result: AggregatedResult):
    """
    Runs merging logic end to end

    Args:
        aggregated_result (aggregated_result): the logs to aggregate
    """
    with tempfile.NamedTemporaryFile() as tmp:
        with open(tmp.name, "wb") as out_file, Writer(out_file) as mcap_writer:
            for message in mcap_message_generator(
                aggregated_result.s3_paths, aggregated_result.names
            ):
                mcap_writer.write_message(
                    topic=message.topic,
                    message=message.proto,
                    log_time=message.timestamp,
                    publish_time=message.timestamp,
                )
        flattened_names = "_".join(aggregated_result.names).replace("/", "_")
        # upload collated log
        # by default upload the log to s3
        s3_path = f"logs/merged/{flattened_names}/{aggregated_result.common_path}/merged_log.mcap"
        logger.info(f"Uploading merged log to: s3://voxel-temp/{s3_path}")
        upload_file(bucket="voxel-temp", s3_path=s3_path, local_path=tmp.name)


def main(args: argparse.Namespace):
    aggregated_result = aggregated_incidents_from_args(args)
    collate_logs(aggregated_result)


def parse_args() -> argparse.Namespace:
    """
    Parses input commandline args

    Returns:
        (argparse.Namespace): the parsed args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3_logs",
        type=str,
        nargs="+",
        help=(
            "The list of s3 logs: e.g. s3://voxel-temp/foo.mcap "
            "s3://voxel-temp/bar.mcap"
        ),
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
