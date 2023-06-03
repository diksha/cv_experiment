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

import tempfile
import threading
import typing
from typing import Any

from loguru import logger
from mcap_protobuf.writer import Writer as ProtobufWriter

from core.utils.aws_utils import get_bucket_path_from_s3_uri, upload_file


class SynchronousLogWriter:
    _instance = None
    _lock = threading.Lock()
    _temp_file = None
    _writer: typing.Optional[ProtobufWriter] = None

    def __init__(self):
        if self._instance is not None:
            raise ValueError("Must call with get_instance")
        self._temp_file = (
            tempfile.NamedTemporaryFile()  # trunk-ignore(pylint/R1732)
        )
        self._fd = open(  # trunk-ignore(pylint/R1732)
            self._temp_file.name, "wb"
        )
        self._writer = ProtobufWriter(self._fd)
        self._objects = set()
        self._finalized_objects = set()
        self._instance = self

    @classmethod
    def get_instance(cls) -> "SynchronousLogWriter":
        """
        Generates instance of the log writer if it doesn't exist

        Returns:
            SynchronousLogWriter: writer
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register(self, node_name: str):
        with self._lock:
            self._objects.add(node_name)

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    def _get_filename(self) -> str:
        """
        Returns the filename of the log file

        Returns:
            str: the filename of the log file
        """
        return self._temp_file.name

    def write(self, topic: str, timestamp_ns: int, protobuf: typing.Any):
        """
        Synchronously writes out the timestamped message to the protobuf

        Args:
            topic (str): the topic to write the protobuf to
            timestamp_ns (int): the current timestamp of the message
            protobuf (typing.Any): the raw protobuf message
        """
        with self._lock:
            self._writer.write_message(
                topic=topic,
                message=protobuf,
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
            )

    def _close(self):
        """
        Closes and deletes the current instance of the
        writer
        """
        self._writer.finish()
        self._fd.close()
        self._instance = None

    def finalize(self, node_name: str) -> typing.Optional[str]:
        """
        Closes and deletes the current instance of the
        writer

        Args:
            node_name (str): the name of the node to finalize

        Returns:
            typing.Optional[str]: the filename of the log file
        """
        with self._lock:
            self._finalized_objects.add(node_name)
            # check if finalized objects are equal to objects
            if self._objects == self._finalized_objects:
                logger.info(f"Uploading logs for {node_name}")
                self._close()
                return self._get_filename()
            logger.info(f"Skipping uploading for {node_name}")
            return None


class ProtoLogger:
    """
    Basic logger that writes to a synchronous mcap log
    """

    def __init__(self, name: str, config: dict, *args, **kwargs) -> None:
        # defensive programming for uuid and log key
        self.video_uuid = (
            config["video_uuid"] if "video_uuid" in config else ""
        )
        self.log_key = config["log_key"] if "log_key" in config else ""
        self.name = name
        self.log_path = f"s3://voxel-temp/logs/{self.log_key}/{self.video_uuid}/PerceptionRun.mcap"
        if self.log_key:
            logger.info(f"[ProtoLogger|{self.name}] Logging enabled")
            logger.info(f"Log path: {self.log_path}")
            self.enabled = True
        else:
            logger.info(f"[ProtoLogger|{self.name}] Logging disabled")
            self.enabled = False

        self.data = []
        self.name = name

        self.synchronous_writer = SynchronousLogWriter.get_instance()
        self.synchronous_writer.register(self.name)
        # Only update if the log doesn't already exist, never overwrite it.

    def log_input(self, serializable: Any) -> None:
        """
        Logs the input of the node to the log file

        Args:
            serializable (Any): the serializable log file

        Raises:
            Exception: if the exception was raised somewhere during the log
                       process when this was enabled
        """
        if not self.enabled:
            return

        try:
            if self.synchronous_writer is None:
                self.synchronous_writer = SynchronousLogWriter.get_instance()
                self.synchronous_writer.register(self.name)
            base_topic = "/".join(
                ["", self.name, serializable.get_topic_name()]
            )
            self.synchronous_writer.write(
                base_topic,
                serializable.get_timestamp_ns(),
                serializable.to_proto(),
            )
        except Exception as exc:
            logger.exception("Unable to serialize input struct")
            raise exc

    def log_output(self, serializable: Any) -> None:
        """
        Logs the output of the node to the log file

        Args:
            serializable (Any): the input struct to serialize

        Raises:
            Exception: if the exception was raised somewhere during the log
                       process when this was enabled
        """
        if not self.enabled:
            return
        try:
            if self.synchronous_writer is None:
                self.synchronous_writer = SynchronousLogWriter.get_instance()
                self.synchronous_writer.register(self.name)
            base_topic = "/" + self.name
            self.synchronous_writer.write(
                base_topic,
                serializable.get_timestamp_ns(),
                serializable.to_proto(),
            )
            annotation_proto = serializable.to_annotation_protos()
            if annotation_proto is not None:
                self.synchronous_writer.write(
                    "/".join([base_topic, "annotations"]),
                    serializable.get_timestamp_ns(),
                    annotation_proto,
                )

        except Exception as exc:
            logger.exception(
                "[WARNING] logger was unable to serialize proto object"
            )
            raise exc

    def finalize(self):
        """
        Closes up all dependencies and uploads the log to cloud storage
        """
        # TODO make this a GCS file handle and write it out periodically instead of
        # all at the end
        if self.enabled:
            file_name = self.synchronous_writer.finalize(self.name)
            if file_name is not None:
                # self.synchronous_writer.close()
                (
                    destination_bucket_name,
                    destination_relative_s3_path,
                ) = get_bucket_path_from_s3_uri(self.log_path)
                upload_file(
                    bucket=destination_bucket_name,
                    local_path=file_name,
                    s3_path=destination_relative_s3_path,
                )
                logger.success(f"Uploaded log to: {self.log_path}")
                # reset writer in case we use it for another log
                SynchronousLogWriter.reset()
            self.synchronous_writer = None
