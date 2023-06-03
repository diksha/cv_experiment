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
import uuid
from fractions import Fraction

import av
import numpy as np
from loguru import logger

from core.execution.nodes.abstract import AbstractNode
from core.utils.actionable_region_utils import get_actionable_region_polygons
from core.utils.aws_utils import (
    separate_bucket_from_relative_path,
    upload_file,
)


class VideoWriterNode(AbstractNode):
    def __init__(self, config):
        self.camera_uuid = config["camera_uuid"]
        self.output_s3_path = config["video_writer"]["output_s3_path"]
        self.local_path = (
            f"/tmp/{str(uuid.uuid4())}.mp4"  # trunk-ignore(bandit/B108)
        )
        self.container = av.open(self.local_path, mode="w")
        self.stream = self.container.add_stream(
            "h264", rate=int(config["camera"]["frame_rate"])
        )
        self.stream.codec_context.time_base = Fraction(1, 1000)
        self.stream.width = config["camera"]["width"]
        self.stream.height = config["camera"]["height"]
        self.stream.pix_fmt = "yuv420p"
        self.only_draw_frames_with_gt = config["video_writer"].get(
            "only_draw_frames_with_gt", False
        )
        self._buffer = {}

    def process_frame_struct(self, pred_frame_struct, gt_frame_struct):
        """Takes a frame struct and draws the corresponding actors on a frame.

        Args:
            pred_frame_struct (frame): frame struct perception output
            gt_frame_struct (frame): labeled ground truth frame struct
        """
        # When a frame struct is recieved we always assume that the corresponding
        # frame is already present in the buffer. This is a reasonable assumption
        # for now because a frame struct cannot be generated without a frame.
        # The downside of this approach is that a few frames might get discarded if
        # the frame is already not present in the buffer but it is a very rare/impossible
        # event currently.

        # This should never be the case.
        if pred_frame_struct is None:
            return
        frame_ms = pred_frame_struct.relative_timestamp_ms

        if (
            gt_frame_struct
            and gt_frame_struct.relative_timestamp_ms != frame_ms
        ):
            logger.error(
                "GT frame struct and Pred frame struct timestamps differ"
            )
            return

        frame = self._buffer.pop(frame_ms, None)
        if frame is None:
            logger.debug(
                "Frame not available in buffer when frame structs were received"
            )
            return

        if gt_frame_struct is not None:
            frame = gt_frame_struct.draw(
                frame,
                label_type="gt",
            )

        if not self.only_draw_frames_with_gt or pred_frame_struct is not None:
            actionable_region = get_actionable_region_polygons(
                self.camera_uuid,
                pred_frame_struct.frame_height,
                pred_frame_struct.frame_width,
            )
            points = None
            if len(actionable_region) != 0:
                points = [[p.x, p.y] for p in actionable_region[0].vertices]
                points = [np.array(points, np.int32)]
            frame = pred_frame_struct.draw(
                frame,
                label_type="pred",
                actionable_region=points,
            )

        if not self.only_draw_frames_with_gt or gt_frame_struct:
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            video_frame.pts = int(frame_ms)

            for packet in self.stream.encode(video_frame):
                self.container.mux(packet)

    def process_frame(self, frame, frame_ms):
        """Adds a frame to a buffer

        Args:
            frame (np.array): frame
            frame_ms (int): timestamp of frame in milliseconds
        """
        # TODO: Move all this to background thread and create a deque which
        # caches the input data in the main cycle.
        frame = np.copy(frame)
        self._buffer[frame_ms] = frame

    def finalize(self):
        """Closes the video and uploads it to GCS."""
        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        bucket, prefix = separate_bucket_from_relative_path(
            self.output_s3_path
        )
        upload_file(bucket, self.local_path, prefix)
        os.remove(self.local_path)
