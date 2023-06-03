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

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import av
import numpy as np

from core.utils.aws_utils import generate_presigned_url


@dataclass
class VideoReaderOutput:
    relative_timestamp_ms: int
    image: np.ndarray


@dataclass
class S3VideoReaderInput:
    video_path_without_extension: str = None
    video_extension: str = "mp4"
    bucket_name: str = "voxel-logs"


class VideoReader(ABC):
    @abstractmethod
    def read(
        self,
        min_frame_difference_ms: int,
        max_frames: typing.Optional[int],
    ) -> typing.Generator[VideoReaderOutput, None, None]:
        """Reads a video and returns the output as a python generator.

        If min_frame_difference_ms is specified then certain frames will be skipped to satisfy
        a minimum time difference between frames.

        If max_frames is specified and is greater than 0, then only those number of frames
        will be returned at maximum.

        Note: This reader doesn't provide any gurantee whether the frames returned will be ordered by time.

        Args:
            min_frame_difference_ms (int): Specifies the minimum time difference in milliseconds between frames.
            max_frames (int): Max number of frames to be returned.

        Yields:
            typing.Generator: Generates images and frame timestamp
        """

    @property
    @abstractmethod
    def width(self) -> int:
        """Get the width of the video.

        Note: Frames at different timestamp in the video can have varying width.
        """

    @property
    @abstractmethod
    def height(self) -> int:
        """Get the height of the video.

        Note: Frames at different timestamp in the video can have varying height.
        """

    @property
    @abstractmethod
    def num_frames(self) -> int:
        """Returns the num of frames in the video."""

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        """Get the average frame rate of the video.

        Note: Ideally this value should not be used for any computations.
        """

    @property
    @abstractmethod
    def video_uuid(self) -> str:
        """Returns the video uuid."""


class S3VideoReader(VideoReader):
    """Video Reader class that provides functionality to read a video from S3."""

    def __init__(self, s3_video_reader_input: S3VideoReaderInput):
        """Init.

        Args:
          s3_video_reader_input: S3VideoReaderInput struct that identifies a video.
        """
        self._validate_input(s3_video_reader_input)
        self._s3_video_reader_input = s3_video_reader_input
        blob_path = (
            f"{self._s3_video_reader_input.video_path_without_extension}."
            f"{self._s3_video_reader_input.video_extension.lstrip('.')}"
        )

        self._video_container = av.open(
            generate_presigned_url(
                self._s3_video_reader_input.bucket_name, filepath=blob_path
            )
        )
        self._video_stream = self._video_container.streams.video[0]

    def _validate_input(
        self, s3_video_reader_input: S3VideoReaderInput
    ) -> None:
        if not s3_video_reader_input.video_path_without_extension:
            raise ValueError("Input doesn't have contain video_blob_path")
        if not s3_video_reader_input.bucket_name:
            raise ValueError("Input doesn't have contain bucket_name")
        if not s3_video_reader_input.video_extension:
            raise ValueError("Input doesn't have contain video_extension")

    @property
    def width(self) -> int:
        return self._video_stream.width

    @property
    def height(self) -> int:
        return self._video_stream.height

    @property
    def num_frames(self) -> int:
        return self._video_stream.frames

    @property
    def frame_rate(self) -> float:
        return self._video_stream.average_rate

    @property
    def video_uuid(self) -> str:
        return self._s3_video_reader_input.video_path_without_extension

    def read(
        self,
        min_frame_difference_ms: int = 0,
        max_frames: typing.Optional[int] = None,
    ) -> typing.Generator[VideoReaderOutput, None, None]:
        if max_frames is not None and max_frames < 1:
            raise RuntimeError(
                f"max_frames should be greater than 0 or None, provided: {max_frames}"
            )

        last_frame_time_ms = -1 * min_frame_difference_ms
        num_frames_returned = 0

        for frame in self._video_container.decode(self._video_stream):
            frame_pts_ms = int(frame.pts * 1000 * frame.time_base)

            # Skip this frame if the time difference isn't enough.
            if frame_pts_ms - last_frame_time_ms < min_frame_difference_ms:
                continue

            # Check whether required max number of frames have already been returned.
            if max_frames is not None and num_frames_returned > max_frames:
                break

            last_frame_time_ms = frame_pts_ms

            image = av.VideoFrame.to_ndarray(frame, format="bgr24")

            yield VideoReaderOutput(
                relative_timestamp_ms=frame_pts_ms,
                image=image,
            )

            num_frames_returned += 1
