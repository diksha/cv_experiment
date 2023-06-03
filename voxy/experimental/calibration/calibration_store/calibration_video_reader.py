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

import cv2

from core.infra.cloud.gcs_utils import get_video_signed_url


class CalibrationVideoReader:
    def __init__(
        self,
        video_format="mp4",
        version="v1",
        bucket="voxel-cameras",
        project="sodium-carving-227300",
    ):
        self.bucket = bucket
        self.project = project
        self.version = version
        self.video_format = video_format
        self.calibration_folder = "calibration"

    def read(self, camera_uuid):
        """
        Returns a CameraCalibration struct containing calibration data for
        """
        relative_path = os.path.join(camera_uuid, self.calibration_folder, self.version)
        signed_url = get_video_signed_url(
            relative_path, bucket=self.bucket, video_format=self.video_format
        )

        vcap = cv2.VideoCapture(signed_url)
        while True:
            _, frame = vcap.read()
            if frame is None:
                break
            yield frame

        vcap.release()
