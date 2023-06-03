#
# Copyright 2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

from typing import List

from scaleapi.tasks import TaskType

from core.labeling.scale.lib.scale_task_retry import ScaleTaskRetryWrapper
from core.labeling.scale.lib.scale_task_wrapper import ScaleTaskWrapper
from core.labeling.scale.registry.registry import ScaleTaskCreatorRegistry
from core.labeling.scale.task_creation.task_creation_base import TaskCreation
from core.structs.data_collection import DataCollectionType
from core.utils.aws_utils import glob_from_bucket


@ScaleTaskCreatorRegistry.register()
class SafetyVestImageAnnotationTask(TaskCreation):
    def __init__(
        self,
        credentials_arn: str,
        batch_name_prefix: str = "",
        dry_run: bool = False,
    ):
        self.project = "safety_vest_image_annotation"
        super().__init__(
            self.project,
            batch_name_prefix=batch_name_prefix,
            credentials_arn=credentials_arn,
            dry_run=dry_run,
        )

    # trunk-ignore(pylint/W0237)
    def create_task(
        self,
        data_collection_uuid: str,
        fps: int = 0,
        generate_hypothesis: bool = False,
    ) -> List[str]:
        """Create Scale task

        Args:
            data_collection_uuid (str): UUID for data collection to label
            fps (int): unused for image task, default 0
            generate_hypothesis (bool): unused for image task, default False

        Returns:
            List[str]: list of scale unique task IDs created
        """
        dir_name = data_collection_uuid
        bucket_name = "voxel-logs"
        task_unique_ids = []
        for image in glob_from_bucket(bucket_name, dir_name, ("jpg", "png")):
            if image.endswith("/"):
                continue
            frame_uuid = image[len(dir_name) + 1 :].rsplit(".")[0]
            unique_id = f"{data_collection_uuid}_safetyvest_{frame_uuid}"
            attachment = f"s3://{bucket_name}/{image}"

            metadata = {
                "video_uuid": data_collection_uuid,
                "relative_path": image,
                "filename": unique_id,  # This is shown in scale UI
                "taxonomy_version": self.get_taxonomy_version(self.taxonomy),
            }
            payload = dict(
                project=self.project,
                batch=self.batch.name,
                attachment=attachment,
                metadata=metadata,
                unique_id=unique_id,
                clear_unique_id_on_error=True,
                geometries=self.taxonomy["geometries"],
                annotation_attributes=self.taxonomy["annotation_attributes"],
            )

            def create_task():
                """
                Create scale task

                Returns:
                    None: should not return anything
                """
                # TODO(twroge): remove this in favor of proper
                #               scale side effects when
                #               scale has the bug fixed on their end:
                # PERCEPTION-2150
                if self.dry_run:
                    return None
                return self.client.create_task(
                    TaskType.ImageAnnotation,
                    **payload,  # trunk-ignore(pylint/W0640)
                )

            def cancel_task():
                """
                Cancel scale task

                Returns:
                    None: should not return anything
                """
                # TODO(twroge): remove this in favor of proper
                #               scale side effects when
                #               scale has the bug fixed on their end:
                # PERCEPTION-2150
                if self.dry_run:
                    return None
                return self.client.cancel_task(
                    ScaleTaskWrapper(
                        self.credentials_arn
                    ).get_task_id_from_unique_id(
                        unique_id, self.project  # trunk-ignore(pylint/W0640)
                    ),
                    True,
                )

            ScaleTaskRetryWrapper(
                task_creation_call=create_task,
                task_cancel_call=cancel_task,
            ).create_task()

            task_unique_ids.append(unique_id)
        return task_unique_ids

    def get_data_collection_type(self) -> DataCollectionType:
        """Get DataCollectionType for task creator
        Returns:
            DataCollectionType: type of data task contains
        """
        return DataCollectionType.IMAGE_COLLECTION
