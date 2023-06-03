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
import os
import threading
from contextlib import ExitStack
from datetime import datetime, timedelta
from typing import Tuple

import numpy
from loguru import logger

from core.execution.nodes.abstract import AbstractNode
from core.infra.cloud.kinesis_utils import KinesisVideoMediaReader
from third_party.closeablequeue import CloseableQueue

FRAME_DROP_LOG_INTERVAL_SECS = timedelta(seconds=5)


class CameraStreamNode(AbstractNode):
    def __init__(self, config, kvs_read_session=None):
        """

        Args:
            config (dict): Nested dictionary for general graph configuration
            kvs_read_session (Optional[boto3.Session], optional):
                boto3 session for reading from KinesisVideoStreams.
                Defaults to None.
        """
        self._camera_uuid = config["camera_uuid"]
        self._camera_arn = config["camera"]["arn"]
        self._target_fps = config["camera"]["fps"]

        self._kvs_read_session = kvs_read_session

        # store a maximum of 1000 frames in the buffer
        self._frame_queue = CloseableQueue.CloseableQueue(maxsize=1000)

        # do a little coalescing on frame drops because they can
        # overflow logs very easily
        self._last_frame_drop_counter = 0
        self._frame_drop_counter = 0
        self._last_frame_drop_message_time = datetime.min

        self._last_frame_timestamp = 0

        self._kinesis_stream = None
        self._exc = None

    def _check_exc(self):
        if self._exc is not None:
            e = self._exc
            self._exc = None
            raise e

    def start(self):
        logger.info("Initializing Kinesis reader")
        self._kinesis_stream = KinesisVideoMediaReader(
            stream_arn=self._camera_arn,
            run_once=False,
            session=self._kvs_read_session,
        )
        threading.Thread(target=self._run, daemon=True).start()

    def get_width(self):
        self._check_exc()
        if self._kinesis_stream is None:
            raise RuntimeError("get_width() before start()")
        return self._kinesis_stream.get_width()

    def get_height(self):
        self._check_exc()
        if self._kinesis_stream is None:
            raise RuntimeError("get_height() before start()")
        return self._kinesis_stream.get_height()

    def get_frame_rate(self):
        self._check_exc()
        if self._kinesis_stream is None:
            raise RuntimeError("get_fps() called before start()")
        return self._kinesis_stream.get_fps()

    def _log_dropped_frame(self):
        self._frame_drop_counter += 1
        now = datetime.now()
        if (
            now - self._last_frame_drop_message_time
        ) > FRAME_DROP_LOG_INTERVAL_SECS:
            self._last_frame_drop_message_time = now
            drop_diff = (
                self._frame_drop_counter - self._last_frame_drop_counter
            )
            self._last_frame_drop_counter = self._frame_drop_counter
            drop_total = self._frame_drop_counter
            logger.warning(
                f"CameraStreamNode dropped {drop_diff} frames since last report, {drop_total} total"
            )

    def _put_frame(self, frame):
        """Puts a frame in the frame queue, dropping a frame off the front of the queue if the queue is full

        Args:
            frame (PyavFrame): frame to add to the queue
        """

        try:
            self._frame_queue.put(frame, block=False)
        except CloseableQueue.Full:
            # the queue is full, try to pop a frame off and then try again
            try:
                # we use a nonblocking get because the queue is small and there is
                # always a chance it is drained between the put and this get
                self._frame_queue.get(block=False)
                self._log_dropped_frame()
            except CloseableQueue.Empty:
                # we ignore an empty queue in case the queue somehow got drained
                pass
            # try again, this time if we see an exception let it bubble up
            self._frame_queue.put(frame, block=False)

    def _run(self):
        try:
            with ExitStack() as stack:
                if self._kinesis_stream is None:
                    raise RuntimeError(
                        "CameraStreamNode._kinesis_stream must not be None"
                    )
                stream = stack.enter_context(self._kinesis_stream)
                stack.callback(self._frame_queue.close)
                for frame in stream:
                    if frame.timestamp_ms() < self._last_frame_timestamp:
                        logger.warning(
                            f"Dropping non-monotonic timestamp {frame.timestamp_ms()}"
                        )
                        continue
                    self._last_frame_timestamp = frame.timestamp_ms()

                    try:
                        # put frame attempts to add a frame to the queue, dropping one off the back
                        # if it is already full
                        self._put_frame(frame)
                    except CloseableQueue.Full:
                        # dropping a frame, write a log message
                        self._log_dropped_frame()
        # trunk-ignore(pylint/W0703)
        except BaseException as e:
            logger.exception(f"Camera Stream Node Run: {e}")
            self._exc = e
            # TODO: remove the following os._exit() once we confirm reraising exceptions works
            #       to stop the process on errors
            # trunk-ignore(pylint/W0212)
            os._exit(1)

    def get_next_frame(self) -> Tuple[numpy.ndarray, int]:
        """Returns the next frame for processing

        Raises:
            RuntimeError: an unrecoverable error has occurred

        Returns:
            Tuple[numpy.ndarray, int]: a tuple containing the next frame and its timestamp in milliseconds
        """
        self._check_exc()
        try:
            pyav_frame = self._frame_queue.get()
        except CloseableQueue.Closed as e:
            raise RuntimeError("CameraStreamNode frame queue is closed") from e

        return pyav_frame.to_ndarray(), pyav_frame.timestamp_ms()

    def get_total_number_of_frames_dropped(self) -> int:
        """Returns the total number of frames dropped

        Returns:
            int: total number of frame dropped
        """
        return self._frame_drop_counter
