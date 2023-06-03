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

import json
import os
import sys

from core.labeling.label_store.label_reader import LabelReader
from core.ml.data.extraction.writers.label_writer import LabelWriter
from core.ml.data.extraction.writers.status_writer import StatusWriter
from core.structs.video import Video


class LabelExtractor:
    def __init__(
        self,
        video_uuid: str,
        label_reader: LabelReader,
        label_writer: LabelWriter,
        status_writer: StatusWriter,
    ):
        self._video_uuid = video_uuid
        self._label_reader = label_reader
        self._label_writer = label_writer
        self._status_writer = status_writer

    def _publish_status(self, exit_code: int) -> None:
        self._status_writer.write_exit_status(exit_code)
        publish_path = os.path.join(
            self._label_writer.root_save_dir, self._video_uuid
        )
        self._status_writer.publish_status(publish_path)

    def _handle_exception(self) -> None:
        exception = sys.exc_info()
        self._status_writer.write_failure(exception)
        self._publish_status(-1)

    def extract_data(self):
        try:
            labels_frame_map = {
                item.relative_timestamp_ms: item
                for item in Video.from_dict(
                    json.loads(self._label_reader.read(self._video_uuid))
                ).frames
            }

            for frame_ms, frame in labels_frame_map.items():
                self._label_writer.generate_labeled_data(
                    frame_ms,
                    frame,
                )

            self._label_writer.write_labels(self._video_uuid)
            self._publish_status(0)

        except Exception:
            self._handle_exception()
            raise
