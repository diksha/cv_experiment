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
from core.calibration.calibrate_store.calibration_reader import (
    CalibrationReader,
)
from core.calibration.calibrate_store.calibration_writer import (
    CalibrationWriter,
)


class CalibrationBackend:
    def __init__(
        self,
        calibration_format="json",
        version="v1",
        project="sodium-carving-227300",
    ):
        calibration_bucket = "voxel-storage"
        videos_bucket = "voxel-logs"
        self._reader = CalibrationReader(
            calibration_format=calibration_format,
            version=version,
            bucket=calibration_bucket,
            project=project,
        )
        self._writer = CalibrationWriter(
            calibration_format=calibration_format,
            version=version,
            bucket=calibration_bucket,
            project=project,
        )
        self._video_reader = CalibrationVideoReader(
            version=version, bucket=videos_bucket, project=project
        )

    def calibration_reader(self):
        return self._reader

    def calibration_writer(self):
        return self._writer

    def video_reader(self):
        return self._video_reader
