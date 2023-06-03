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

# Example run:  ./bazel run core/labeling/scale/runners:camera_config_scale_task --
# --camera_uuid innovate_manufacturing/knoxville/0004/cha
import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Callable

import shortuuid
from loguru import logger
from scaleapi.tasks import TaskType

from core.labeling.scale.lib.scale_client import get_scale_client
from core.labeling.scale.lib.scale_task_retry import ScaleTaskRetryWrapper
from core.labeling.scale.lib.scale_task_wrapper import ScaleTaskWrapper
from core.labeling.tools.pull_kinesis_feed import get_frame_from_kinesis
from core.utils.aws_utils import does_blob_exist, upload_cv2_imageobj_to_s3

SCALE_LIVE_CREDENTIAL = (
    "arn:aws:secretsmanager:us-west-2:203670452561:"
    "secret:scale_credentials-WHUbar"
)
SCALE_TEST_CREDENTIAL = (
    "arn:aws:secretsmanager:us-west-2:203670452561:"
    "secret:scale_test_credentials-E77TFB"
)


class CameraConfigAnnotationTask:
    _TAXONOMY_PATH = "core/labeling/scale/task_creation/taxonomies"

    def __init__(self, camera_uuids: list, frame_time: str, is_test: bool):
        logger.info(f"Initializing camera config task creation {camera_uuids}")
        self.project = "camera_config"
        self.camera_uuids = camera_uuids
        self.frame_time = frame_time
        credentials_arn = (
            SCALE_TEST_CREDENTIAL if is_test else SCALE_LIVE_CREDENTIAL
        )
        self.client = get_scale_client(credentials_arn)
        self.credentials_arn = credentials_arn
        taxonomy_path = os.path.join(
            self._TAXONOMY_PATH, f"{self.project}.json"
        )
        with open(taxonomy_path, "r", encoding="UTF-8") as taxonomy_file:
            self.taxonomy = json.load(taxonomy_file)
        self.batch = self.client.create_batch(
            project=self.project,
            batch_name=f"batch_{os.path.commonpath(self.camera_uuids)}_{shortuuid.uuid()}",
        )
        self.is_test = is_test

    def get_frame_and_upload_to_s3(self, camera_uuid: str) -> str:
        """Get frame for camera uuid and upload to s3
        video-logs/camera_config

        Args:
            camera_uuid (str): uuid of the camera

        Raises:
            RuntimeError: frame from camera exists

        Returns:
            str: s3 path of the frame
        """
        logger.info("Getting image and uploading to s3")
        s3_path = (
            f"s3://voxel-logs/camera_config/{camera_uuid}.png"
            if not self.is_test
            else f"s3://voxel-temp/camera_config/{camera_uuid}.png"
        )
        if not self.is_test:
            if does_blob_exist(s3_path):
                raise RuntimeError(
                    f"Image from {camera_uuid} already exists."
                    "Please remove before creating new scale task"
                )
        image = get_frame_from_kinesis(camera_uuid, self.frame_time)
        upload_cv2_imageobj_to_s3(s3_path, image)
        return s3_path

    def create_task(self) -> None:
        """Create a task for camera config

        Raises:
            RuntimeError: task creation failed
        """
        successful_camera_uuids = []
        failed_camera_uuids = []
        for camera_uuid in self.camera_uuids:
            try:
                s3_path = self.get_frame_and_upload_to_s3(camera_uuid)
                camera_uuid = (
                    camera_uuid
                    if not self.is_test
                    else f"{camera_uuid}_test_{shortuuid.uuid()}"
                )
                payload = dict(
                    project=self.project,
                    batch=self.batch.name,
                    attachment=s3_path,
                    metadata={
                        "camera_uuid": camera_uuid,
                        "filename": camera_uuid,
                    },
                    unique_id=camera_uuid,
                    clear_unique_id_on_error=True,
                    geometries=self.taxonomy["geometries"],
                    annotation_attributes=self.taxonomy[
                        "annotation_attributes"
                    ],
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
                successful_camera_uuids.append(camera_uuid)
            except Exception:  # trunk-ignore(pylint/W0703)
                logger.exception(f"Failed to create task for {camera_uuid}")
                failed_camera_uuids.append(camera_uuid)
            logger.info(
                f"Scale task created for {successful_camera_uuids} "
                f"and failed for {failed_camera_uuids}"
            )
        self.batch.finalize()
        if failed_camera_uuids:
            raise RuntimeError(
                f"Failed to create scale tasks for {failed_camera_uuids}"
            )


def create_camera_uuids(input_args) -> list:
    """Using the input args, outputs a list of camera_uuids based on:
    1. A given list using args.camera_uuids
    2. Obtaining camera uuids given the args.org_zone and args.custom_camera_numbers

    Args:
        input_args (_type_): input parser arguments

    Raises:
        ValueError: If neither camera_uuids or org/zone is provided with camera range/numbers

    Returns:
        list: camera_uuids
    """
    if not input_args.camera_uuids:
        input_args.camera_uuids = []
        if not (input_args.org_zone and input_args.custom_camera_numbers):
            raise ValueError(
                "Please enter either camera_uuids or org_zone with camera numbers"
            )
        for num in input_args.custom_camera_numbers:
            input_args.camera_uuids.append(
                f"{input_args.org_zone}/{num:04d}/cha"
            )
    return input_args.camera_uuids


def parse_args() -> argparse.Namespace:
    """Parses arguments for the binary

    Returns:
        argparse.Namespace: Command line arguments passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--camera_uuids",
        metavar="C",
        nargs="+",
        type=str,
        help="camera_uuids to ingest",
    )
    parser.add_argument(
        "-o",
        "--org_zone",
        metavar="O",
        type=str,
        help="Required when using custom camera numbers",
    )
    parser.add_argument(
        "-n",
        "--custom_camera_numbers",
        metavar="N",
        nargs="+",
        type=int,
        help="Required with org/zone: Custom camera numbers to ingest",
    )
    parser.add_argument(
        "-f",
        "--frame_time",
        metavar="f",
        type=str,
        default=(datetime.now() - timedelta(minutes=30)).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        help="Frame time example: 2022-12-23 14:12:10",
    )
    parser.add_argument(
        "-t",
        "--is_test",
        action="store_true",
        help="Create test task",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.camera_uuids = create_camera_uuids(args)
    CameraConfigAnnotationTask(
        args.camera_uuids, args.frame_time, args.is_test
    ).create_task()
