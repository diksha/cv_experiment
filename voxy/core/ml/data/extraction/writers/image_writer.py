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

import numpy as np

from core.ml.data.extraction.utils.data_extraction_helpers import (
    generate_image_filename,
)
from core.utils.aws_utils import upload_cv2_imageobj_to_s3


class ImageWriter:
    def __init__(
        self,
        bucket: str,
        relative_path: str,
    ):
        self._root_save_dir = os.path.join(bucket, relative_path)

    def write_image(
        self,
        video_uuid: str,
        frame_timestamp_ms: int,
        frame: np.ndarray,
    ) -> None:
        file_name = generate_image_filename(frame_timestamp_ms)
        save_path = f"s3://{self._root_save_dir}/{video_uuid}/{file_name}"
        upload_cv2_imageobj_to_s3(save_path, frame)

    @property
    def root_save_dir(self):
        return self._root_save_dir
