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
import sys

from core.ml.data.extraction.writers.image_writer import ImageWriter
from core.ml.data.extraction.writers.status_writer import StatusWriter
from core.utils.video_reader import VideoReader


class ImageExtractor:
    def __init__(
        self,
        video_reader: VideoReader,
        image_writer: ImageWriter,
        status_writer: StatusWriter,
        frame_timestamp_filter: set = None,
    ):
        self._video_reader = video_reader
        self._image_writer = image_writer
        self._status_writer = status_writer
        self._frame_timestamp_filter = frame_timestamp_filter

    def _publish_status(self, exit_code: int) -> None:
        self._status_writer.write_exit_status(exit_code)
        publish_path = os.path.join(
            self._image_writer.root_save_dir, self._video_reader.video_uuid
        )
        self._status_writer.publish_status(publish_path)

    def _handle_exception(self) -> None:
        exception = sys.exc_info()
        self._status_writer.write_failure(exception)
        self._publish_status(-1)

    def _should_extract_frame(self, frame_timestamp_ms: int) -> bool:
        if self._frame_timestamp_filter:
            return frame_timestamp_ms in self._frame_timestamp_filter
        return True

    def extract_data(self):
        try:
            video_uuid = self._video_reader.video_uuid

            for video_reader_op in self._video_reader.read():
                frame_ms = video_reader_op.relative_timestamp_ms
                frame = video_reader_op.image
                if self._should_extract_frame(frame_ms):
                    self._image_writer.write_image(
                        video_uuid,
                        frame_ms,
                        frame,
                    )
            self._publish_status(0)

        except Exception:
            self._handle_exception()
            raise
