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
from typing import Callable, List

from loguru import logger

from core.labeling.scale.lib.converters.batch_task_utils import (
    get_completed_batch_video_map,
)
from core.labeling.scale.lib.converters.converter_base import (
    Converter,
    VideoData,
)
from core.labeling.scale.registry.registry import ScaleLabelConverterRegistry


@ScaleLabelConverterRegistry.register()
class ImageCollectionAnnotationConverter(Converter):
    """
    DISCLAIMER: THIS UPDATES VIDEO STRUCT WITH ONLY PARTICULAR ACTORS(Door,Person).
    If same video_uuid is used for another task, this class overwrites
    the prior task data
    """

    def __init__(
        self,
        completion_before_date: str,
        completion_after_date: str,
        project_name: str,
        consumable_labels_fn: Callable,
        credentials_arn: str,
    ):
        self._project_name = project_name
        self.consumable_labels_fn = consumable_labels_fn
        super().__init__(
            completion_before_date,
            completion_after_date,
            self._project_name,
            credentials_arn,
        )

    def convert_and_upload(self) -> List[VideoData]:
        """
        Main conversion function for all queried batches
        Returns:
            List[VideoData]: list of video uuids and metadata that have been converted

        """
        logger.info(
            f"Starting to convert tasks and upload for {self._project_name}"
        )
        all_converted_video_uuids = []

        batch_map = get_completed_batch_video_map(
            self._project_name,
            self.completion_after_date,
            self.completion_before_date,
            self.credentials_arn,
        )
        for _, video_task_map in batch_map.items():
            for video_uuid, video_tasks in video_task_map.items():
                consumable_labels = self.consumable_labels_fn(
                    video_uuid, video_tasks
                )
                upload_successful = super().upload_consumable_labels_to_s3(
                    video_uuid,
                    consumable_labels.to_dict(),
                )
                if upload_successful:
                    all_converted_video_uuids.append(
                        VideoData(
                            video_tasks[0].metadata,
                            video_uuid,
                        )
                    )
        return all_converted_video_uuids
