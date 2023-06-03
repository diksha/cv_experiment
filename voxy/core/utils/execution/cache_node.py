##
## Copyright 2022 Voxel Labs, Inc.
## All rights reserved.
##
## This document may not be reproduced, republished, distributed, transmitted,
## displayed, broadcast or otherwise exploited in any manner without the express
## prior written permission of Voxel Labs, Inc. The receipt or possession of this
## document does not convey any rights to reproduce, disclose, or distribute its
## contents, or to manufacture, use, or sell anything that it may describe, in
## whole or in part.
##

import typing

from loguru import logger
from mcap_protobuf.reader import read_protobuf_messages


class LogCache:
    def __init__(self, log_file: str):
        logger.info(f"Log from: {log_file}")
        self.log_reader_generator = iter(
            read_protobuf_messages(source=log_file, log_time_order=True)
        )
        self.last_timestamp_ns = -float("inf")
        self.topic_map = {}

    def seek(self, node_name: str, timestamp_ms: int) -> typing.Any:
        """
        Seeks to a time in the log file and returns
        the message for the topic at the specific time

        Args:
            node_name (str): the name of the node to return the protobuf for
            timestamp_ms (int): timestamp to sample the log. Timestamps must be monotonically given
                                for efficient log seeking, otherwise an error is thrown

        Raises:
            ValueError: if the timestamps given are not monotonic

        Returns:
            typing.Any: optionally gives the value at the time requested.
                        Returns None if it is not found in the log
        """
        timestamp_ns = int(timestamp_ms * 1e6)
        if timestamp_ns < self.last_timestamp_ns:
            raise ValueError(
                "Timestamps added were not monotonic! Last timestamp was: "
                f"{self.last_timestamp_ns} and current timestamp is {timestamp_ns}"
            )
        if timestamp_ns == self.last_timestamp_ns:
            return self.topic_map.get(node_name)

        current_timestamp_ns = self.last_timestamp_ns
        self.last_timestamp_ns = timestamp_ns
        # clear the topic map
        self.topic_map = {}

        while current_timestamp_ns <= timestamp_ns:
            try:
                message = next(self.log_reader_generator)
            except StopIteration:
                logger.info("Reached the end of the log file")
                return None
            current_timestamp_ns = message.log_time
            self.topic_map[message.topic.lstrip("/")] = message.proto_message
        return self.topic_map.get(node_name)
