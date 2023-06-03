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

from loguru import logger

from core.labeling.cvat.client import CVATClient
from core.utils.aws_utils import (
    get_secret_from_aws_secret_manager,
    upload_directory_to_s3,
)

_CVAT_HOST = "cvat.voxelplatform.com"
_CAMERA_CONFIG_CVAT_PROJECT_ID = 10
_PROD_CVAT_PROJECT_ID = 8


def export_all_cvat_data(project_id: int):
    """Exports all cvat data and store it in
    voxel-perception/gcloud_backup/cvat_data

    Args:
        project_id (int): id of project
    """
    cvat_client = CVATClient(
        _CVAT_HOST,
        get_secret_from_aws_secret_manager("CVAT_CREDENTIALS"),
        project_id=project_id,
    )
    for task in cvat_client.all_tasks_generator():
        logger.info(f'Task name : {task["name"]}, Task id : {task["id"]}')
        with tempfile.TemporaryDirectory() as tmpdirname:
            cvat_client.download_cvat_labels(
                task["name"], "cvat_images", False, tmpdirname
            )
            cvat_client.download_task_data(
                task["id"], tmpdirname, task["name"]
            )
            upload_directory_to_s3(
                bucket="voxel-perception",
                local_directory=tmpdirname,
                s3_directory="gcloud_backup/cvat_data",
            )


if __name__ == "__main__":
    logger.info("Getting tasks for prod project")
    export_all_cvat_data(_PROD_CVAT_PROJECT_ID)
    logger.info("Getting tasks for camera config project")
    export_all_cvat_data(_CAMERA_CONFIG_CVAT_PROJECT_ID)
