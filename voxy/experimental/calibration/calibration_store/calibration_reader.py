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
import datetime
import os
import pickle

from core.infra.cloud.gcs_utils import (
    read_from_gcs,
    video_uuid_to_gcs_path,
)
from core.structs.calibrated_camera import CameraCalibration

# TODO (harishma): Test with CameraCalibration struct when available


class CalibrationReader:
    def __init__(
        self,
        calibration_format="pkl",
        version="v1",
        bucket="voxel-storage",
        project="sodium-carving-227300",
    ):
        self.bucket = bucket
        self.project = project
        self.version = version
        self.calibration_format = calibration_format

    def read(self, camera_uuid):
        """
        Returns a CameraCalibration struct containing calibration data for
        """
        relative_path = os.path.join(self.version, camera_uuid)
        full_gcs_path = "gs://{}/{}.{}".format(
            self.bucket, relative_path, self.calibration_format
        )
        calibration_data = read_from_gcs(full_gcs_path)
        if not calibration_data:
            print("No calibration data found for camera ", camera_uuid)
            return None
        return CameraCalibration.from_pickle(calibration_data)
