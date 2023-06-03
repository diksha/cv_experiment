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
import json
import os

from core.calibration.calibration_store.calibration_reader import CalibrationReader
from core.infra.cloud.gcs_utils import dump_to_gcs


class CalibrationWriter(CalibrationReader):
    def __init__(
        self,
        calibration_format="pkl",
        version="v1",
        bucket="voxel-storage",
        project="sodium-carving-227300",
    ):
        super().__init__(version, bucket, project)
        self.bucket = bucket
        self.project = project
        self.version = version
        self.calibration_format = calibration_format

    def write_labels(self, labels, video_uuid):
        relative_filepath = self._insert_version_into_path(video_uuid)
        full_gcs_path = "gs://{}/{}.{}".format(
            self.bucket, relative_filepath, self.label_format
        )
        return dump_to_gcs(
            full_gcs_path, pickle.dump(labels), "application/json", self.project
        )

    def _insert_version_into_path(self, path):
        return path + version.replace("v", "")
