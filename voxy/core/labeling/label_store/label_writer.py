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

from core.labeling.label_store.label_reader import LabelReader
from core.utils.aws_utils import upload_fileobj_to_s3


class LabelWriter(LabelReader):
    def __init__(
        self,
        label_format="json",
        version="v1",
        bucket="voxel-consumable-labels",
        project="sodium-carving-227300",
    ):
        super().__init__(version, bucket, project)
        self.bucket = bucket
        self.project = project
        self.version = version
        self.label_format = label_format

    def write_labels(self, labels, video_uuid):
        """
        Write labels to S3
        Args:
            labels (dict): labels to write
            video_uuid (str): video uuid writing to
        """
        relative_filepath = self._insert_version_into_path(video_uuid)
        full_aws_path = (
            f"s3://{self.bucket}/{relative_filepath}.{self.label_format}"
        )
        upload_fileobj_to_s3(
            full_aws_path,
            json.dumps(labels).encode("utf-8"),
            "application/json",
        )

    def _insert_version_into_path(self, path):
        return os.path.join(self.version, path)
