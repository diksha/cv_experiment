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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

from botocore.exceptions import ConnectionError as BotoCoreConnectionError
from botocore.exceptions import HTTPClientError
from loguru import logger

from core.utils.aws_utils import upload_fileobj_to_s3
from core.utils.voxel_decorators import retry_handler


@dataclass
class VideoData:
    metadata: Dict[str, object]
    video_uuid: str


class Converter(ABC):
    def __init__(
        self,
        completion_before_date: str,
        completion_after_date: str,
        project: str,
        credentials_arn: str,
    ) -> None:
        self.project = project
        self.completion_before_date = completion_before_date
        self.completion_after_date = completion_after_date
        self.credentials_arn = credentials_arn
        logger.info(f"Initializing converter task for {project}")

    @abstractmethod
    def convert_and_upload(self) -> List[VideoData]:
        """
        Abstract function to convert and upload labels
        Returns:
            List[VideoData]: list of video uuids and metadata that have been converted
        """
        raise NotImplementedError("Converter must implement convert method")

    @retry_handler(
        exceptions=(
            BotoCoreConnectionError,
            HTTPClientError,
        ),
        max_retry_count=2,
        retry_delay_seconds=1,
        backoff_factor=3,
    )
    def upload_consumable_labels_to_s3(
        self, video_uuid: str, consumable_labels: dict
    ) -> bool:
        """
        Uploads consumable labels over to s3://voxel-consumable-labels
        Args:
            video_uuid (str): video_uuid to of labels being uploaded
            consumable_labels (dict): consumable labels of video being uploaded
        Returns:
            bool: Upload status
        """
        full_s3_path = f"s3://voxel-consumable-labels/v1/{video_uuid}.json"
        upload_bytes = json.dumps(consumable_labels).encode("utf-8")
        s3_dump_successful = upload_fileobj_to_s3(
            full_s3_path, upload_bytes, "application/json"
        )
        if s3_dump_successful is False:
            logger.error(f"Failed to upload labels for {video_uuid} to s3")
            return False
        logger.info(f"Uploaded labels for {video_uuid} to s3")
        return True
