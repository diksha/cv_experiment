#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
import json
import random
import tempfile
from pathlib import Path

import docker
from docker.models.containers import Container
from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client.models.dataset_type import (
    DatasetType,
)
from lightly.openapi_generated.swagger_client.models.datasource_purpose import (
    DatasourcePurpose,
)
from loguru import logger

from core.ml.data.curation.lib.lightly_worker import wait_on_worker
from core.ml.data.curation.voxel_lightly_utils import LIGHTLY_TOKEN_ARN
from core.utils.aws_utils import get_secret_from_aws_secret_manager

# The directory within the Lightly container that will hold the
# output data
_WITHIN_WORKER_OUTPUT_DIRECTORY = "/home/output_dir"

# trunk-ignore(pylint/C0301)
LIGHTLY_DOCKER_IMAGE_PATH = "203670452561.dkr.ecr.us-west-2.amazonaws.com/lightly/boris-250909/lightly/worker:latest"


LIGHTLY_DELEGATED_ACCESS_ROLE_ARN_TOKEN_ARN = (
    # trunk-ignore(bandit/B105)
    "arn:aws:secretsmanager:us-west-2:203670452561:"
    "secret:LIGHTLY_DELEGATED_ACCESS_ROLE_ARN-DCdCUt"
)
LIGHTLY_DELEGATED_ACCESS_EXTERNAL_ID_TOKEN_ARN = (
    # trunk-ignore(bandit/B105)
    "arn:aws:secretsmanager:us-west-2:203670452561:"
    "secret:LIGHTLY_DELEGATED_ACCESS_EXTERNAL_ID-tCrQIr"
)


def get_worker_config() -> dict:
    """Get worker config for yolo subsampling

    Returns:
        dict: worker config
    """
    return {
        "worker": {"force_start": False},
        "enable_training": False,
        "method": "coreset",
        "stopping_condition": {"min_distance": 0.1, "n_samples": 250000},
        "datasource": {"bypass_verify": True},
    }


def docker_container(lightly_token, worker_id) -> Container:
    """
    Clean starts a docker container

    Args:
        lightly_token (str): lightly token
        worker_id (str): lightly worker id. Reuse right now, since creating new
        worker_id is a noop

    Raises:
        RuntimeError: Unable to start docker container

    Returns:
        Container: container running lightly worker
    """
    logger.info("Cleaning and starting up docker container..")
    docker_client = docker.from_env()
    _container_output_dir = Path(tempfile.gettempdir()) / Path(
        # trunk-ignore(bandit/B311)
        f"lightly-output-{random.randint(0, 2**64)}"
    )
    # Pull lightly image
    image_found = False
    for image in docker_client.images.list():
        if LIGHTLY_DOCKER_IMAGE_PATH in image.tags:
            image_found = True
            break
    if not image_found:
        container_registry = LIGHTLY_DOCKER_IMAGE_PATH
        logger.info(f"Container not found, pulling from {container_registry}")
        try:
            docker_client.images.pull(container_registry)
        except RuntimeError as exc:
            raise RuntimeError(
                """If you see a 'no basic auth credentials' error, run: \n
                ./tools/aws ecr get-login-password --region us-west-2 |
                docker login --username AWS --password-stdin
                203670452561.dkr.ecr.us-west-2.amazonaws.com"""
            ) from exc
    # Kill all containers running lightly
    for container in docker_client.containers.list():
        if LIGHTLY_DOCKER_IMAGE_PATH in container.image.tags:
            logger.info(f"Killing: {container}")
            container.kill()
    logger.info("Running docker container")
    return docker_client.containers.run(
        LIGHTLY_DOCKER_IMAGE_PATH,
        f"token={lightly_token} worker.worker_id={worker_id}",
        volumes={
            _container_output_dir.as_posix(): {
                "bind": _WITHIN_WORKER_OUTPUT_DIRECTORY,
                "mode": "rw",
            }
        },
        detach=True,
        **{
            "runtime": "nvidia",
            "shm_size": "1gb",
        },
    )


def yolo_subsample():
    """Subsample yolo data using lightly"""
    lightly_token = json.loads(
        get_secret_from_aws_secret_manager(LIGHTLY_TOKEN_ARN)
    )["1"]
    lightly_client = ApiWorkflowClient(token=lightly_token)
    lightly_client.create_new_dataset_with_unique_name(
        "yolo_downsample",
        DatasetType.IMAGES,
    )
    delegated_access_role_arn = get_secret_from_aws_secret_manager(
        LIGHTLY_DELEGATED_ACCESS_ROLE_ARN_TOKEN_ARN
    )
    delegated_access_external_id = get_secret_from_aws_secret_manager(
        LIGHTLY_DELEGATED_ACCESS_EXTERNAL_ID_TOKEN_ARN
    )
    input_path = "s3://voxel-lightly-input/yolo/lightly_downsample/"
    output_path = "s3://voxel-lightly-output/yolo/lightly_downsample/"
    lightly_client.set_s3_delegated_access_config(
        resource_path=input_path,
        region="us-west-2",
        role_arn=delegated_access_role_arn,
        external_id=delegated_access_external_id,
        purpose=DatasourcePurpose.INPUT,
    )
    # Output bucket
    lightly_client.set_s3_delegated_access_config(
        resource_path=output_path,
        region="us-west-2",
        role_arn=delegated_access_role_arn,
        external_id=delegated_access_external_id,
        purpose=DatasourcePurpose.LIGHTLY,
    )
    dataset_id = lightly_client.dataset_id
    logger.info(f"Dataset id {dataset_id}")
    worker_id = lightly_client.register_compute_worker()
    logger.info(f"Worker id is {worker_id}")
    container = docker_container(lightly_token, worker_id)
    worker_config = get_worker_config()
    scheduled_run_id = lightly_client.schedule_compute_worker_run(
        worker_config=worker_config
    )
    wait_on_worker(
        lightly_client,
        scheduled_run_id,
        container.logs(stream=True),
    )


if __name__ == "__main__":
    yolo_subsample()
