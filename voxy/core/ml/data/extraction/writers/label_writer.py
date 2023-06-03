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

from core.ml.data.extraction.label_generation.label_generator import (
    LabelGenerator,
)
from core.structs.frame import Frame


class LabelWriter:
    def __init__(
        self,
        label_generator: LabelGenerator,
        bucket: str,
        relative_path: str,
    ):
        self._root_save_dir = os.path.join(bucket, relative_path)
        self._label_generator = label_generator

    def generate_labeled_data(
        self,
        frame_timestamp_ms: int,
        frame: Frame,
    ) -> None:
        self._label_generator.process_frame(
            frame_timestamp_ms,
            frame,
        )

    def write_labels(
        self,
        video_uuid: str,
    ) -> None:
        save_dir = f"{self._root_save_dir}/{video_uuid}"
        self._label_generator.publish_labels(
            save_dir,
        )

    @property
    def root_save_dir(self):
        return self._root_save_dir
