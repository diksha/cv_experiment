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
import fractions
from dataclasses import dataclass
from typing import IO, Optional, Union

import av
import numpy
from loguru import logger


class PyavError(Exception):
    pass


@dataclass
class PyavFrame(av.frame.Frame):
    """Wraps the PyAV stream and frame data

    This adds some convenience methods for timestamps, frame data, etc
    to avoid computational errors
    """

    frame: av.video.frame.VideoFrame

    def to_ndarray(self) -> numpy.ndarray:
        """Converts the frame image data to a numpy array in bgr24

        Returns:
            numpy.ndarray: numpy array of the frame data
        """
        return av.VideoFrame.to_ndarray(self.frame, format="bgr24")

    def timestamp_ms(self) -> int:
        """Returns the frame presentation timestamp in milliseconds

        Returns:
            int: frame presentation timestamp in milliseconds
        """
        # use the stream time base because it is actually more accurate
        # than the frame time base strangely.
        return int(self.frame.time_base * 1000 * self.frame.pts)

    def width(self) -> int:
        """Returns the frame width in pixels

        Returns:
            int: frame width in pixels
        """
        return self.frame.width

    def height(self) -> int:
        """Returns the frame height in pixels

        Returns:
            int: frame height in pixels
        """
        return self.frame.height


# TODO: Update to support more inputs than just a fifo for input
class PyavDecoder:
    """PyavDecoder decodes a video file written via write and produces the frames via get_frame

    Args:
        fps        (int): sets the sampling rate for frames in frames per second
        max_frames (int): sets the maximum number of frames in the queue
                          before frames will be dropped. the queue will
                          be unbounded if this value is 0 (default)
    """

    def __init__(
        self,
        file: Union[str, IO[bytes]],
        fps: Optional[float] = None,
        # trunk-ignore(pylint/W0622)
        format: Optional[str] = None,
    ):
        logger.info(f"Initializing decoder for {file}")
        self._fps_sampled: Optional[float] = (
            1 / fps if fps is not None else None
        )
        self._last_timestamp_ms: Optional[int] = None
        self._timenext: int = 0

        try:
            self._container = av.open(
                file, metadata_errors="strict", format=format
            )
        except av.FFmpegError as e:
            raise PyavError("failed to open input for decoding") from e
        self._decoder = self._container.decode(video=0)
        self._container.streams.video[0].thread_type = "AUTO"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._container.close()

    def __iter__(self):
        return self

    def __next__(self) -> PyavFrame:
        frame = self.get_frame()
        if frame is None:
            raise StopIteration
        return frame

    def close(self) -> None:
        """Must be called to clean up the decoder instance"""
        self._container.close()

    def _stream(self) -> av.stream.Stream:
        return self._container.streams.video[0]

    def get_width(self) -> int:
        """Returns the video width

        Returns:
            int: video width in pixels
        """
        return self._stream().width

    def get_height(self) -> int:
        """Returns the video height

        Returns:
            int: video height in pixels
        """
        return self._stream().height

    def get_last_timestamp_ms(self) -> Optional[int]:
        """Returns the last frame timestamp returned by get_frame

        Returns:
            Optional[int]: last timestamp in milliseconds, or None of no frames have been produced yet
        """
        return self._last_timestamp_ms

    def get_fps(self) -> fractions.Fraction:
        """Returns fps as a Fraction

        Returns:
            fractions.Fraction: fps
        """
        return self._stream().average_rate

    def get_frame(self) -> Optional[av.frame.Frame]:
        """Returns the next frame from the decoder if available, and None when the end of the input has been reached.

        Raises:
            PyavError: an unexpected decoder failure occurred

        Returns:
            av.frame.Frame: This will be either the next available frame or None if the end of the stream has been reached.
        """
        try:
            for frame in self._decoder:
                pyav_frame = PyavFrame(frame=frame)
                current_timestamp_ms = pyav_frame.timestamp_ms()

                if (
                    self._last_timestamp_ms is not None
                    and current_timestamp_ms <= self._last_timestamp_ms
                ):
                    logger.warning(
                        f"Dropping non-monotonically increasing ts: {current_timestamp_ms}"
                    )
                    continue

                if self._fps_sampled is None:
                    self._last_timestamp_ms = current_timestamp_ms
                    return pyav_frame

                if current_timestamp_ms >= self._timenext:
                    self._timenext = current_timestamp_ms + int(
                        self._fps_sampled * 1000
                    )
                    self._last_timestamp_ms = current_timestamp_ms
                    return pyav_frame
        except av.FFmpegError as e:
            raise PyavError("ffmpeg decode error") from e
        return None
