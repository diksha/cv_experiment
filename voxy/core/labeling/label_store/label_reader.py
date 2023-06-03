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

from loguru import logger

from core.infra.cloud.gcs_utils import read_from_gcs
from core.utils.aws_utils import (
    does_blob_exist,
    get_blobs_from_bucket,
    read_decoded_bytes_from_s3,
)


class LabelReader:
    def __init__(
        self,
        label_format="json",
        version="v1",
        bucket="voxel-consumable-labels",
        project="sodium-carving-227300",
        provider="s3",
    ):
        self.bucket = bucket
        self.project = project
        self.version = version
        self.label_format = label_format
        self.provider = provider

    @staticmethod
    def get_all_consumable_label_video_uuids(
        label_version: str,
        label_format: str = "json",
    ) -> list:
        consumable_label_video_uuids = []
        for blob in get_blobs_from_bucket(
            "voxel-consumable-labels", label_version
        ):
            relative_path, extension = os.path.splitext(blob.key)
            if label_format in extension:
                video_uuid = ("/").join(relative_path.split("/")[1:])
                consumable_label_video_uuids.append(video_uuid)
        return consumable_label_video_uuids

    def read(self, video_uuid):
        relative_path = os.path.join(self.version, video_uuid)
        if self.provider == "s3":
            full_s3_path = (
                f"s3://{self.bucket}/{relative_path}.{self.label_format}"
            )
            if not does_blob_exist(full_s3_path):
                logger.warning(f"Unable to get s3 labels for {video_uuid}")
                return None
            return read_decoded_bytes_from_s3(full_s3_path)
        # TODO(diksha): Remove gcs path after voxel-raw-labels migration
        full_gcs_path = (
            f"gs://{self.bucket}/{relative_path}.{self.label_format}"
        )
        return read_from_gcs(full_gcs_path)
