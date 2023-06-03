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
from typing import Optional, Tuple

import av
import numpy
from loguru import logger
from tqdm import tqdm

from core.execution.nodes.abstract import AbstractNode
from core.execution.utils.frame_queue import FrameQueue
from core.utils.aws_utils import generate_presigned_url


class VideoStreamNode(AbstractNode):
    def __init__(self, config):
        self._min_frame_difference_ms = config["camera"][
            "min_frame_difference_ms"
        ]
        self.video_source_bucket = config.get("video_stream", {}).get(
            "video_source_bucket"
        )
        # As we have 10 minute video chunks and approx 10-11 fps
        # using the min_frame_difference_ms of 85ms, we will be around
        # 6000-7000 frames maximum, therefore keep the queue size
        # for that to prevent video buffering fully before perception
        # can start.
        self._frame_queue = FrameQueue(7500)
        self._video_completed = False
        self._video_container = av.open(
            generate_presigned_url(
                self.video_source_bucket, f'{config["video_uuid"]}.mp4'
            )
        )
        self._video_stream = self._video_container.streams.video[0]
        self._total_number_of_frames = self._video_stream.frames
        self._max_frames_to_process = min(
            config.get("max_frame_count", self._total_number_of_frames),
            self._total_number_of_frames,
        )
        self._video_uuid = config["video_uuid"]
        disable_pbar = True
        self._progress_bar_submitted_for_processing = tqdm(
            total=self._max_frames_to_process,
            desc=f"Frames submitted for processing for {self._video_uuid}",
            disable=disable_pbar,
        )
        self._progress_bar_frames_retrieved_for_processing = tqdm(
            total=self._max_frames_to_process,
            desc=f"Frames retrieved for processing for {self._video_uuid}",
            disable=disable_pbar,
        )

    def get_width(self):
        return self._video_stream.width

    def get_height(self):
        return self._video_stream.height

    def get_frame_rate(self):
        # Don't return Fraction as it's not json serializable.
        return float(self._video_stream.average_rate)

    def is_video_complete(self):
        return self._video_completed

    def run(self):
        last_frame_time_ms = -1 * self._min_frame_difference_ms
        try:
            number_of_frames_submitted_for_processing = 0
            for frame in self._video_container.decode(self._video_stream):
                frame_ms = int(frame.time * 1000)
                if (
                    frame_ms - last_frame_time_ms
                    < self._min_frame_difference_ms
                ):
                    continue

                last_frame_time_ms = frame_ms
                image = av.VideoFrame.to_ndarray(frame, format="bgr24")
                self._frame_queue.put(image, frame_ms)

                number_of_frames_submitted_for_processing += 1
                self._progress_bar_submitted_for_processing.update(1)

                if (
                    number_of_frames_submitted_for_processing
                    >= self._max_frames_to_process
                ):
                    break

            # Ensure we set this here when we reach end of video because
            # above we might not have processed self._max_frames_to_process
            # as we skip frames using self._min_frame_difference_ms.
            self._video_completed = True
        # trunk-ignore(pylint/W0703)
        except Exception as e:
            logger.exception(f"Video Stream Node Run: {e}")
            # trunk-ignore(pylint/W0212)
            os._exit(1)

    def get_next_frame(self) -> Tuple[Optional[numpy.ndarray], int]:
        return self._frame_queue.get()
