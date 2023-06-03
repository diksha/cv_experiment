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

import tempfile
from typing import List

import av
from scaleapi.tasks import TaskType

from core.labeling.constants import VOXEL_VIDEO_LOGS_BUCKET
from core.labeling.scale.hypothesis_generation.videoplayback_hypothesis import (
    annotations_hypothesis,
)
from core.labeling.scale.lib.scale_task_retry import ScaleTaskRetryWrapper
from core.labeling.scale.lib.scale_task_wrapper import ScaleTaskWrapper
from core.labeling.scale.registry.registry import ScaleTaskCreatorRegistry
from core.labeling.scale.task_creation.task_creation_base import TaskCreation
from core.structs.data_collection import DataCollectionType
from core.utils.aws_utils import download_to_file


@ScaleTaskCreatorRegistry.register()
class VideoPlaybackAnnotationTask(TaskCreation):
    def __init__(
        self,
        credentials_arn: str,
        batch_name_prefix: str = "",
        dry_run: bool = False,
    ):
        self.project = "video_playback_annotation"
        super().__init__(
            self.project,
            batch_name_prefix=batch_name_prefix,
            credentials_arn=credentials_arn,
            dry_run=dry_run,
        )

    def _get_fps(self, video_uuid):
        with tempfile.NamedTemporaryFile() as tmp_file:
            video_path = download_to_file(
                VOXEL_VIDEO_LOGS_BUCKET, f"{video_uuid}.mp4", tmp_file.name
            )
            fps = int(
                av.open(video_path, metadata_errors="strict")
                .streams.video[0]
                .average_rate
            )
        return fps

    def create_task(
        self, video: str, fps: float, generate_hypothesis: bool = False
    ) -> List[str]:
        """Create a labeling task for the given video at the given FPS

        Args:
            video: The UUID of the video to be labeled
            fps: The frames-per-second of the video for labeling
            generate_hypothesis: Whether to generate a hypothesis for the video

        Returns:
            A list with one element: the UUID of the task created
        """
        video_gcs_path = f"s3://{VOXEL_VIDEO_LOGS_BUCKET}/{video}.mp4"

        frame_rate = 1
        if fps:
            video_fps = self._get_fps(video)
            frame_rate = int(video_fps / fps)

        hypothesis = None
        if generate_hypothesis:
            (
                s3_path,
                frame_rate,
                _,
            ) = annotations_hypothesis.AnnotationHypothesis(video).process()
            hypothesis = {"annotations": {"url": s3_path}}

        # Internally constant frame rate converts our video to constant
        # frame rate video and send us keyframes instead.
        payload = dict(
            project=self.project,
            batch=self.batch.name,
            attachment_type="video",
            attachment=video_gcs_path,
            unique_id=f"{video}",
            clear_unique_id_on_error=True,
            convert_to_constant_frame_rate_video=True,
            constant_frame_rate=frame_rate,
            metadata={
                "original_video_path": f"s3://{VOXEL_VIDEO_LOGS_BUCKET}/{video}.mp4",
                "video_uuid": video,
                "filename": video,  # This is shown in Scale UI
                "taxonomy_version": self.get_taxonomy_version(self.taxonomy),
            },
            geometries=self.taxonomy["geometries"],
            events_to_annotate=self.taxonomy["events_to_annotate"],
            annotation_attributes=self.taxonomy["annotation_attributes"],
            hypothesis=hypothesis,
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
                TaskType.VideoPlaybackAnnotation,
                **payload,
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
                ).get_task_id_from_unique_id(video, self.project),
                True,
            )

        ScaleTaskRetryWrapper(
            task_creation_call=create_task,
            task_cancel_call=cancel_task,
        ).create_task()
        return [video]

    def get_data_collection_type(self) -> DataCollectionType:
        """Get DataCollectionType for task creator
        Returns:
            DataCollectionType: type of data task contains
        """
        return DataCollectionType.VIDEO
