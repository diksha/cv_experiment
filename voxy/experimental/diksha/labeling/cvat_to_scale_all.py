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

import argparse

# Example run:  ./bazel run core/labeling/scale/runners:camera_config_scale_task --
# --camera_uuid innovate_manufacturing/knoxville/0004/cha
import json
import os
from typing import Callable

import shortuuid
from loguru import logger
from scaleapi.tasks import TaskType

from core.labeling.cvat.client import CVATClient
from core.labeling.scale.lib.scale_client import get_scale_client
from core.labeling.scale.lib.scale_task_retry import ScaleTaskRetryWrapper
from core.labeling.scale.lib.scale_task_wrapper import ScaleTaskWrapper
from core.utils.aws_utils import get_secret_from_aws_secret_manager
from experimental.diksha.labeling.cvat_to_scale_camera_config import (
    get_annotations_from_cvat,
)


class CameraConfigAnnotationTask:
    _TAXONOMY_PATH = "core/labeling/scale/task_creation/taxonomies"

    def __init__(self, credentials_arn):
        self.project = "camera_config"
        self.client = get_scale_client(credentials_arn)
        self.credentials_arn = credentials_arn
        taxonomy_path = os.path.join(
            self._TAXONOMY_PATH, f"{self.project}.json"
        )
        with open(taxonomy_path, "r", encoding="UTF-8") as taxonomy_file:
            self.taxonomy = json.load(taxonomy_file)
        self.batch = self.client.create_batch(
            project=self.project,
            batch_name=f"batch_cvat_to_scale_{shortuuid.uuid()}",
        )

    def create_task(self) -> None:
        """Create a task for camera config

        Raises:
            RuntimeError: task creation failed
        """

        cvat_client = CVATClient(
            "cvat.voxelplatform.com",
            get_secret_from_aws_secret_manager("CVAT_CREDENTIALS"),
            project_id=10,
        )

        failed_cameras = []
        for task in cvat_client.all_tasks_generator():
            camera_uuid = os.path.relpath(task["name"], "camera_config/")

            annotations, s3_path = get_annotations_from_cvat(camera_uuid)
            logger.info(f"Image path {s3_path}")
            try:
                payload = dict(
                    project=self.project,
                    batch=self.batch.name,
                    attachment=s3_path,
                    metadata={
                        "camera_uuid": camera_uuid,
                        "filename": camera_uuid,
                    },
                    unique_id=f"{camera_uuid}",
                    clear_unique_id_on_error=True,
                    geometries=self.taxonomy["geometries"],
                    annotation_attributes=self.taxonomy[
                        "annotation_attributes"
                    ],
                    hypothesis={"annotations": annotations},
                )

                def create_task() -> Callable:
                    """Create task function for camera_config

                    Returns:
                        Callable: create scale task function
                    """
                    return self.client.create_task(
                        TaskType.ImageAnnotation,
                        **payload,  # trunk-ignore(pylint/W0640)
                    )

                def cancel_task() -> Callable:
                    """Cancel scale task function for camera config

                    Returns:
                        Callable: Cancel scale task callable
                    """
                    return self.client.cancel_task(
                        ScaleTaskWrapper(
                            self.credentials_arn
                        ).get_task_id_from_unique_id(
                            camera_uuid,  # trunk-ignore(pylint/W0640)
                            self.project,
                        ),
                        True,
                    )

                ScaleTaskRetryWrapper(
                    task_creation_call=create_task,
                    task_cancel_call=cancel_task,
                ).create_task()
            except Exception:  # trunk-ignore(pylint/W0703)
                logger.exception(f"Failed to create task for {camera_uuid}")
                failed_cameras.append(camera_uuid)
        self.batch.finalize()
        logger.info(f"Create scale tasks, failed for {failed_cameras}")


def parse_args() -> argparse.Namespace:
    """Parses arguments for the binary

    Returns:
        argparse.Namespace: Command line arguments passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--credentials_arn",
        type=str,
        default=(
            "arn:aws:secretsmanager:us-west-2:203670452561:"
            "secret:scale_credentials-WHUbar"
        ),
        help="Credetials arn",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CameraConfigAnnotationTask(args.credentials_arn).create_task()
