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

from copy import deepcopy
from typing import List

import cv2
from loguru import logger
from scaleapi.tasks import TaskType

from core.incidents.utils import CameraConfig
from core.labeling.scale.lib.scale_task_retry import ScaleTaskRetryWrapper
from core.labeling.scale.lib.scale_task_wrapper import ScaleTaskWrapper
from core.labeling.scale.registry.registry import ScaleTaskCreatorRegistry
from core.labeling.scale.task_creation.task_creation_base import TaskCreation
from core.structs.actor import ActorCategory, get_track_uuid
from core.structs.attributes import RectangleXYXY
from core.structs.data_collection import DataCollectionType
from core.utils.aws_utils import upload_cv2_imageobj_to_s3
from core.utils.video_reader import S3VideoReader, S3VideoReaderInput
from core.utils.video_utils import get_camera_uuid

CROPPED_IMAGE_BUCKET = "voxel-datasets"
CROPPED_IMAGE_PREFIX = "derived/voxel/doors"


@ScaleTaskCreatorRegistry.register()
class DoorClassificationTask(TaskCreation):
    def __init__(
        self,
        credentials_arn: str,
        batch_name_prefix="",
        dry_run=False,
    ):
        self.project = "door_state_classification"
        super().__init__(
            self.project,
            batch_name_prefix=batch_name_prefix,
            credentials_arn=credentials_arn,
            dry_run=dry_run,
        )

    def create_task(
        self, video: str, fps: float, generate_hypothesis: bool = False
    ) -> List[str]:
        """Submit scale tasks for labeling the given video at the provided frames-per-second

        Args:
            video(str): The UUID of the video to be labeled
            fps(float): The frames-per-second of the video to be labeled
            generate_hypothesis(bool): unused for door task, default False

        Returns:
            A list of unique ids for every submitted task

        Raises:
            Exception: if no doors are found in the camera config
        """
        video_reader_input = S3VideoReaderInput(
            video_path_without_extension=video
        )
        video_reader = S3VideoReader(video_reader_input)

        camera_uuid = get_camera_uuid(video)
        camera_config = None
        task_unique_ids = []

        if fps:
            min_frame_difference_ms = 1000 / fps
        else:
            min_frame_difference_ms = 0
        for video_reader_op in video_reader.read(
            min_frame_difference_ms=min_frame_difference_ms
        ):
            frame_ms = video_reader_op.relative_timestamp_ms
            original_frame = video_reader_op.image
            if not camera_config:
                camera_config = CameraConfig(
                    camera_uuid,
                    original_frame.shape[0],
                    original_frame.shape[1],
                )
                doors = camera_config.doors

            if not doors:
                raise Exception("No doors were found in the camera config")
            for _, door in enumerate(doors):
                frame = deepcopy(original_frame)

                rect = RectangleXYXY.from_polygon(door.polygon)
                cv2.rectangle(
                    frame,
                    (rect.top_left_vertice.x, rect.top_left_vertice.y),
                    (rect.bottom_right_vertice.x, rect.bottom_right_vertice.y),
                    (0, 255, 0),
                    3,
                )
                full_frame_filename = (
                    f"{door.orientation}_{door.door_id}_{frame_ms}.jpg"
                )
                full_frame_s3_path = (
                    f"s3://{CROPPED_IMAGE_BUCKET}/{CROPPED_IMAGE_PREFIX}/"
                    f"{video}/{full_frame_filename}"
                )
                upload_cv2_imageobj_to_s3(full_frame_s3_path, frame)
                attachments = [
                    {"type": "image", "content": full_frame_s3_path},
                ]
                unique_id = f"{video}_{str(door.door_id)}_{frame_ms}"
                task_unique_ids.append(unique_id)
                metadata = {
                    "video_uuid": video,
                    "door_id": door.door_id,
                    "track_uuid": get_track_uuid(
                        camera_uuid,
                        str(door.door_id),
                        ActorCategory.DOOR,
                    ),
                    "timestamp_ms": frame_ms,
                    "door_orientation": door.orientation,
                    "height": frame.shape[0],
                    "width": frame.shape[1],
                    "camera_config_version": camera_config.version,
                    "door_type": door.door_type,
                    "door_polygon": door.polygon.to_dict(),
                    "filename": unique_id,  # This is shown in scale UI
                    "taxonomy_version": self.get_taxonomy_version(
                        self.taxonomy
                    ),
                }
                logger.info(f"Creating task with unique id: {unique_id}")
                payload = dict(
                    project=self.project,
                    batch=self.batch.name,
                    attachments=attachments,
                    metadata=metadata,
                    unique_id=unique_id,
                    clear_unique_id_on_error=True,
                    fields=self.taxonomy["fields"],
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
                    return (
                        self.client.create_task(
                            TaskType.TextCollection,
                            **payload,  # trunk-ignore(pylint/W0640)
                        )
                        if not self.dry_run
                        else None
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
                    return (
                        self.client.cancel_task(
                            ScaleTaskWrapper(
                                self.credentials_arn
                            ).get_task_id_from_unique_id(
                                unique_id,  # trunk-ignore(pylint/W0640)
                                self.project,
                            ),
                            True,
                        )
                        if not self.dry_run
                        else None
                    )

                ScaleTaskRetryWrapper(
                    task_creation_call=create_task,
                    task_cancel_call=cancel_task,
                ).create_task()
        logger.info(f"task created for {video}")
        return task_unique_ids

    def get_data_collection_type(self) -> DataCollectionType:
        """Get DataCollectionType for task creator
        Returns:
            DataCollectionType: type of data task contains
        """
        return DataCollectionType.IMAGE_COLLECTION
