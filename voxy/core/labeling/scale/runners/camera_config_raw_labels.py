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

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta

import cv2
from loguru import logger
from scaleapi.tasks import TaskStatus

from core.labeling.scale.lib.scale_task_wrapper import ScaleTaskWrapper
from core.utils.aws_utils import (
    download_to_file,
    get_bucket_path_from_s3_uri,
    upload_fileobj_to_s3,
)

SCALE_LIVE_CREDENTIAL = (
    "arn:aws:secretsmanager:us-west-2:203670452561:"
    "secret:scale_credentials-WHUbar"
)
SCALE_TEST_CREDENTIAL = (
    "arn:aws:secretsmanager:us-west-2:203670452561:"
    "secret:scale_test_credentials-E77TFB"
)


def camera_config_task_to_raw_labels(
    completion_before_date: str, lookback_days: float, is_test: bool
):
    """Scale camera config task move to voxel-raw-labels

    Args:
        completion_before_date (str): Move scale task to voxel-raw-labels
        lookback_days (float): Number of days old task to look up
        is_test (bool): scale credentials for test
    """
    end_date = datetime.strptime(completion_before_date, "%Y-%m-%d %H:%M:%S")
    duration = timedelta(days=lookback_days)
    completion_after_date = (end_date - duration).strftime("%Y-%m-%d %H:%M:%S")
    credentials_arn = (
        SCALE_TEST_CREDENTIAL if is_test else SCALE_LIVE_CREDENTIAL
    )
    task_list = ScaleTaskWrapper(credentials_arn).get_active_tasks(
        project_name="camera_config",
        updated_after=completion_after_date,
        updated_before=completion_before_date,
        status=TaskStatus("completed"),
    )
    for task in task_list:
        bucket, path = get_bucket_path_from_s3_uri(task.params["attachment"])
        task_dict = task.as_dict()
        with tempfile.NamedTemporaryFile() as temp:
            download_to_file(bucket, path, temp.name)
            image = cv2.imread(temp.name)
            height, width, _ = image.shape
            task_dict["image_shape"] = {"height": height, "width": width}
        camera_config_path = (
            "s3://voxel-raw-labels/camera_config/"
            if not is_test
            else "s3://voxel-temp/camera_config/"
        )
        s3_path = os.path.join(
            camera_config_path,
            f'{task.metadata["camera_uuid"]}.json',
        )
        logger.info(f'Uploading {task.metadata["camera_uuid"]} to {s3_path}')
        upload_fileobj_to_s3(s3_path, json.dumps(task_dict).encode("utf-8"))


def parse_args() -> argparse.Namespace:
    """Parses arguments for the binary

    Returns:
        argparse.Namespace: Command line arguments passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--completion_before_date",
        type=str,
        default=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )
    parser.add_argument(
        "--lookback_days",
        type=float,
        default=2.0,
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
    camera_config_task_to_raw_labels(
        args.completion_before_date, args.lookback_days, args.is_test
    )
