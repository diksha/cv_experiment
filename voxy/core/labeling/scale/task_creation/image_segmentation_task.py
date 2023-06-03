from typing import List

from scaleapi.tasks import TaskType

from core.labeling.scale.lib.scale_task_retry import ScaleTaskRetryWrapper
from core.labeling.scale.lib.scale_task_wrapper import ScaleTaskWrapper
from core.labeling.scale.registry.registry import ScaleTaskCreatorRegistry
from core.labeling.scale.task_creation.task_creation_base import TaskCreation
from core.structs.data_collection import DataCollectionType
from core.utils.aws_utils import upload_cv2_imageobj_to_s3
from core.utils.video_reader import S3VideoReader, S3VideoReaderInput

IMAGES_BUCKET = "voxel-datasets"
IMAGES_PREFIX = "derived/voxel/spills"


@ScaleTaskCreatorRegistry.register()
class ImageSegmentationTask(TaskCreation):
    """Wrapper class for Scale's Image Segmentation task creation API"""

    def __init__(
        self,
        credentials_arn: str,
        project: str = "image_segmentation_prod",
        batch_name_prefix: str = "",
        dry_run: bool = False,
    ):
        self.project = project
        super().__init__(
            self.project,
            batch_name_prefix=batch_name_prefix,
            credentials_arn=credentials_arn,
            dry_run=dry_run,
        )

    def extract_images_and_metadata(
        self, video_uuid: str, min_frame_difference_ms: int = 0
    ) -> dict:
        """Given a video extract frames that are at least min_frame_difference_ms apart.
        Args:
            video_uuid: uuid referencing video from which to extract frames
            min_frame_difference_ms: minimum time different between extracted frames

        Returns:
            A dictionary of image path to metadata pairings.
        """
        extracted_image_paths = {}

        video_reader_input = S3VideoReaderInput(
            video_path_without_extension=video_uuid
        )
        video_reader = S3VideoReader(video_reader_input)

        for video_reader_op in video_reader.read(
            min_frame_difference_ms=min_frame_difference_ms
        ):
            frame_ms = video_reader_op.relative_timestamp_ms
            frame = video_reader_op.image
            full_frame_filename = f"{video_uuid}_{frame_ms}.jpg"
            full_frame_s3_path = (
                f"s3://{IMAGES_BUCKET}/{IMAGES_PREFIX}/"
                f"{video_uuid}/{full_frame_filename}"
            )
            upload_cv2_imageobj_to_s3(full_frame_s3_path, frame)
            metadata = {
                "original_video_path": full_frame_s3_path,
                "video_uuid": video_uuid,
                "timestamp": frame_ms,
                "filename": video_uuid,
            }
            extracted_image_paths[full_frame_s3_path] = metadata

        return extracted_image_paths

    def create_task(
        self, video: str, fps: float = 0, generate_hypothesis: bool = False
    ) -> List[str]:
        """Create Image Segmentation tasks in Scale from images extracted
        from a videos. All tasks genetated are part of the same batch.
        Args:
            video: uuid referencing video from which to
                         extract images and create tasks
            fps: the rate of sample of images in the given video
            generate_hypothesis: whether to generate hypothesis for the task

        Returns:
            Task ids
        """
        if fps:
            min_frame_difference_ms = 1000 / fps
        else:
            min_frame_difference_ms = 0

        extracted_images_metadata = self.extract_images_and_metadata(
            video, min_frame_difference_ms
        )
        task_unique_ids = []
        for (
            extracted_image_path,
            extracted_image_metadata,
        ) in extracted_images_metadata.items():
            frame_ms = str(extracted_image_metadata["timestamp"])
            unique_id = f"{video}_{frame_ms}"
            payload = dict(
                project=self.project,
                batch=self.batch.name,
                attachment_type="image",
                attachment=extracted_image_path,
                unique_id=unique_id,
                clear_unique_id_on_error=True,
                metadata=extracted_image_metadata,
                labels=self.taxonomy["labels"],
                allow_unlabeled=True,
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
                    TaskType.SegmentAnnotation,
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
