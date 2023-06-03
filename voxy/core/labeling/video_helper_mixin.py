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

from core.labeling.constants import VOXEL_VIDEO_LOGS_BUCKET
from core.utils.video_reader import S3VideoReader, S3VideoReaderInput


class VideoHelperMixin:
    def get_frame_timestamp_ms_map(
        self, video_uuid: str, metadata_bucket: str = VOXEL_VIDEO_LOGS_BUCKET
    ) -> dict:
        """Generates a map of frame numbers to frame timestamps in milliseconds."""
        frame_map = {}
        video_reader_input = S3VideoReaderInput(
            video_path_without_extension=video_uuid
        )
        video_reader = S3VideoReader(video_reader_input)
        for frame_index, video_reader_op in enumerate(video_reader.read()):
            frame_ms = video_reader_op.relative_timestamp_ms
            frame = video_reader_op.image
            frame_map[frame_index] = (frame_ms, frame.shape[0], frame.shape[1])
        return frame_map
