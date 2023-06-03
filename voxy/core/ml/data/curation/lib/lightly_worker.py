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
import json
import os
import random
import shutil

# trunk-ignore-all(bandit/B404)
import subprocess
import sys
import tempfile
import time
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import CalledProcessError
from typing import Dict, List

import docker
import sematic
from docker.models.containers import Container
from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client.models.dataset_type import (
    DatasetType,
)
from loguru import logger

from core.infra.sematic.shared.resources import GPU_4CPU_16GB_1x
from core.ml.data.curation.voxel_lightly_utils import (
    LIGHTLY_TOKEN_ARN,
    LightlyVideoFrameSequence,
    get_or_create_existing_dataset_by_name,
)
from core.utils.aws_utils import get_secret_from_aws_secret_manager

# The directory within the Lightly container that will hold the
# output data
_WITHIN_WORKER_OUTPUT_DIRECTORY = "/home/output_dir"

_WITHIN_WORKER_ENTRYPOINT = "/home/boris/onprem-docker/entrypoint.sh"

_WITHIN_WORKER_PYTHON_DIR = "/opt/conda/envs/env/bin/"
# too many instance attributes

LIGHTLY_DOCKER_IMAGE_PATH = "lightly/worker:2.6.5"


class LightlyWorker:
    """Downsample as a service

    Given a list of images or videos or crops in s3 bucket and config for
    stopping condition, gives a list of images and sequences in output.
    """

    def __init__(
        self,
        dataset_name: str,
        input_directory: str,
        output_directory: str,
        config: dict,
        dataset_type: str,
        notify: bool,
        existing_dataset_id: typing.Optional[str] = None,
    ) -> None:
        """Initializes lightly worker

        Args:
            dataset_name (str): name of the dataset, if dataset exists,
            will create unique name by appending a number.
            input_directory (str): directory in voxel-lightly-input consisting
            images or videos to input
            output_directory (str): directory in voxel-lightly-output where results
            are to be sent
            config (dict): the configuration to be forwarded to the lightly api
            dataset_type (str): type of dataset either "images", "videos" or "crops"
        """
        logger.info("Initializing Lightly Worker")
        self.dataset_name = dataset_name
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.dataset_type = getattr(DatasetType, dataset_type)
        self.config = config
        lightly_token = json.loads(
            get_secret_from_aws_secret_manager(LIGHTLY_TOKEN_ARN)
        )["1"]
        self.lightly_client = ApiWorkflowClient(token=lightly_token)
        self.worker_assignment_tag = str(uuid.uuid4())
        worker_id = self.lightly_client.register_compute_worker(
            name=self.worker_assignment_tag,
            labels=[self.worker_assignment_tag],
        )
        self._container_output_dir = Path(tempfile.gettempdir()) / Path(
            # trunk-ignore(bandit/B311)
            f"lightly-output-{random.randint(0, 2**64)}"
        )
        logger.info(f"Mounting output to {self._container_output_dir}")
        self.container = self._clean_start_docker_container(
            lightly_token, worker_id
        )
        self.notify = notify
        self.existing_dataset_id = existing_dataset_id
        logger.info("Initialized Lightly Worker")

    def _clean_start_docker_container(
        self, lightly_token: str, worker_id: str
    ) -> Container:
        """Clean starts a docker container

        Args:
            lightly_token (str): lightly token
            worker_id (str): lightly worker id. Reuse right now, since creating new
            worker_id is a noop

        Returns:
            Container: container running lightly worker

        Raises:
            RuntimeError: if the container cannot be pulled
        """
        logger.info("Cleaning and starting up docker container..")
        docker_client = docker.from_env()
        # Pull lightly image
        image_found = False
        for image in docker_client.images.list():
            if LIGHTLY_DOCKER_IMAGE_PATH in image.tags:
                image_found = True
                break
        if not image_found:
            container_registry = LIGHTLY_DOCKER_IMAGE_PATH
            logger.info(
                f"Container not found, pulling from {container_registry}"
            )
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
                self._container_output_dir.as_posix(): {
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

    def _cleanup_worker(self):
        """Cleans up the docker container"""
        logger.info("Cleaning up worker")
        self.container.stop()
        self.container.remove()

    def __enter__(self) -> "LightlyWorker":
        """Work with "with" python notion

        Returns:
            LightlyWorker: lightly worker
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            shutil.rmtree(self._container_output_dir)
        except PermissionError:
            # sometimes the output directory could not be deleted because
            # of permission issues (/tmp/...) so this catches that for local runs
            logger.exception(
                f"Could not remove output directory: "
                f"{self._container_output_dir}"
            )
        self._cleanup_worker()

    def run(self) -> List[LightlyVideoFrameSequence]:
        """Runs the lightly container and pushes the output to output directory.

        Returns:
            List[LightlyVideoFrameSequence]: lightly video frame sequences.
        """
        logger.info("Running Lightly Worker")
        logger.info(f"Creating or getting Lightly Dataset {self.dataset_name}")
        get_or_create_existing_dataset_by_name(
            self.lightly_client,
            self.existing_dataset_id,
            self.dataset_name,
            self.input_directory,
            self.output_directory,
            self.dataset_type,
            notify=self.notify,
        )

        logger.info("Scheduling Run")
        scheduled_run_id = self.lightly_client.schedule_compute_worker_run(
            runs_on=[self.worker_assignment_tag], **self.config
        )
        wait_on_worker(
            self.lightly_client,
            scheduled_run_id,
            self.container.logs(stream=True),
        )
        if (
            self.config.get("worker_config", {}).get(
                "selected_sequence_length", 0
            )
            > 1
        ):
            lightly_output = output_frame_sequence(
                self.lightly_client, scheduled_run_id
            )
        else:
            lightly_output = output_sampled_images(self.lightly_client)
        logger.info(f"Lightly output {lightly_output}")
        return lightly_output


def wait_on_worker(
    lightly_client: ApiWorkflowClient,
    scheduled_run_id: str,
    log_line_source: typing.Optional[typing.Iterable[bytes]] = None,
    lightly_process: typing.Optional[subprocess.Popen] = None,
):
    """Wait on a scheduled run while streaming logs.

    Args:
        lightly_client: the Lightly Api client
        scheduled_run_id: the id of the run to wait on
        log_line_source: an optional iterable yielding log lines
        lightly_process: an optional reference to the process running lightly
            to query if the process has crashed
    """
    logger.info("Entering wait")

    def state_status():
        """
        Basic job to stream the current worker status.

        Prints the current worker state and the message from the worker

        Raises:
            RuntimeError: Lightly API reports run failed
        """
        logger.info("Starting status loop")
        for run_info in lightly_client.compute_worker_run_info_generator(
            scheduled_run_id=scheduled_run_id
        ):
            logger.info(
                f"Compute worker state {run_info.state} with message {run_info.message}"
            )
        # trunk-ignore(pylint/W0631)
        if run_info.ended_successfully():
            logger.info("Lightly API reports successful status")
        else:
            logger.info("Lightly API reports run failed")
            # trunk-ignore(pylint/W0212)
            os._exit(1)

    def container_log_streamer():
        """
        Basic container log streamer job for the docker container.

        Streams the current logs from the docker container and formats it
        to the console. To suppress, set the log level to something higher
        than DEBUG
        """
        for line in log_line_source:
            logger.debug(line.decode("utf-8").strip())

    executor = ThreadPoolExecutor()
    state_status_future = executor.submit(state_status)
    log_streamer_future = None
    if log_line_source is not None:
        log_streamer_future = executor.submit(container_log_streamer)

    # busy wait
    logger.info("Starting busy wait")
    while True:
        if state_status_future.done():
            logger.info("Lightly API signaled completed run")
            state_status_future.result()
            break
        if (
            lightly_process is not None
            and (status_code := lightly_process.poll()) is not None
        ):
            if status_code == 0:
                logger.info("Lightly process completed, exiting")
            else:
                logger.info(
                    f"Lightly process crashed, status_code, {status_code}"
                )
            state_status_future.cancel()
            break
        time.sleep(1)
    logger.info("Worker is done")

    # we cancel the streamer since this will hold up the process when the
    # state status streamer finishes
    if log_streamer_future is not None and not log_streamer_future.done():
        log_streamer_future.cancel()
    executor.shutdown(wait=False)


def output_frame_sequence(
    lightly_client: ApiWorkflowClient,
    scheduled_run_id: str,
) -> List[LightlyVideoFrameSequence]:
    """If the worker was performing sequence selection, get the frame sequence

    Must be executed AFTER calling 'run' and before this LightlyWorker's context
    is exited.

    Args:
        lightly_client (ApiWorkflowClient): lightly client
        scheduled_run_id (str): id of the lightly run

    Returns:
        List of LightlyVideoFrameSequences specifying sequences identified by Lightly
    """

    # Download sequence information
    run = lightly_client.get_compute_worker_run_from_scheduled_run(
        scheduled_run_id=scheduled_run_id
    )
    with tempfile.NamedTemporaryFile() as temp:
        lightly_client.download_compute_worker_run_sequence_information(
            run=run, output_path=temp.name
        )
        with open(temp.name, "r", encoding="utf8") as file:
            decoded_json = json.load(file)

            return [
                LightlyVideoFrameSequence(**frame_sequence)
                for frame_sequence in decoded_json
            ]


def output_sampled_images(
    lightly_client: ApiWorkflowClient,
) -> List[LightlyVideoFrameSequence]:
    """Gets a list of sampled images from output directory
    Args:
        lightly_client (ApiWorkflowClient): lightly client

    Returns:
        List[LightlyVideoFrameSequence]: specifying sampled images identified by Lightly
    """
    filenames_and_read_urls = (
        lightly_client.export_filenames_and_read_urls_by_tag_name(
            tag_name="initial-tag"  # name of the tag in the dataset
        )
    )
    video_frame_map = {}
    for filename in filenames_and_read_urls:
        video_name = "-".join(filename["fileName"].split("-")[0:-2])
        if not video_frame_map.get(video_name):
            video_frame_map[video_name] = []
        video_frame_map[video_name].append(filename["fileName"])
    result = []
    logger.info(f"Video frame map {video_frame_map}")
    for key, value in video_frame_map.items():
        result.append(
            LightlyVideoFrameSequence(
                video_name=key,
                frame_names=value,
                frame_indices=[],
                frame_timestamps_pts=[],
                frame_timestamps_sec=[],
            )
        )
    return result


# TODO(twroge): make these standard resources when lightly has the OOM issue resolved
# trunk-ignore-begin(pylint/W9011,pylint/W9015)
@sematic.func(
    resource_requirements=GPU_4CPU_16GB_1x,
    base_image_tag="lightly-image",
    standalone=True,
)
def run_lightly_worker(
    dataset_id: str,
    dataset_name: str,
    input_directory: str,
    output_directory: str,
    config: Dict[str, object],
    dataset_type: str = "VIDEOS",
    notify: bool = False,
) -> List[LightlyVideoFrameSequence]:
    """# Perform sequence selection with Lightly and get the frame sequence

    ## Parameters
    - **dataset_id**:
        If of the dataset.
    - **dataset_name**:
        Name of the dataset. If dataset exists, will create unique name by
        appending a number.
    - **input_directory**:
        Directory in voxel-lightly-input consisting of images or videos to input
    - **output_directory**:
        directory in voxel-lightly-output where results are to be sent
    - **config**:
        configuration for lightly run
    - **dataset_type**:
        dataset type
    - **notify**
        notify flag

    ## Returns
    List of LightlyVideoFrameSequence specifying sequences identified by Lightly

    ## Raises
    :raises CalledProcessError: if lightly process crashes
    """
    # trunk-ignore-end(pylint/W9011,pylint/W9015)
    if os.path.exists(_WITHIN_WORKER_ENTRYPOINT):
        new_env = dict(**os.environ)
        updated_path = f"{_WITHIN_WORKER_PYTHON_DIR}:{new_env.get('PATH')}"
        new_env["PATH"] = updated_path
        new_env["PYTHONPATH"] = ""

        # should switch to secret manager once permissions are solved
        lightly_token = json.loads(
            get_secret_from_aws_secret_manager(LIGHTLY_TOKEN_ARN)
        )["1"]
        lightly_client = ApiWorkflowClient(token=lightly_token)
        worker_assignment_tag = str(uuid.uuid4())
        worker_id = lightly_client.register_compute_worker(
            name=worker_assignment_tag, labels=[worker_assignment_tag]
        )
        dataset_id = get_or_create_existing_dataset_by_name(
            lightly_client,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            input_dir=input_directory,
            output_dir=output_directory,
            dataset_type=getattr(DatasetType, dataset_type),
            notify=notify,
        )
        cmd = [
            "/bin/bash",
            _WITHIN_WORKER_ENTRYPOINT,
            f"token={lightly_token}",
            f"worker.worker_id={worker_id}",
        ]
        # trunk-ignore(bandit/B603)
        with subprocess.Popen(
            cmd,
            env=new_env,
            cwd="/home/boris",
        ) as child_process:
            os.environ["LIGHTLY_WORKER_ID"] = worker_id
            try:
                scheduled_run_id = lightly_client.schedule_compute_worker_run(
                    runs_on=[worker_assignment_tag],
                    **config,
                )
            # trunk-ignore(pylint/W0718)
            except Exception:
                logger.exception("Unable to schedule run")
                sys.exit(1)
            logger.info(f"Scheduled: {scheduled_run_id}")
            wait_on_worker(
                lightly_client, scheduled_run_id, lightly_process=child_process
            )
            if status_code := child_process.poll() not in (None, 0):
                raise CalledProcessError(status_code, cmd)
            child_process.terminate()
            child_process.wait(timeout=60)
            if (
                config.get("worker_config", {}).get(
                    "selected_sequence_length", 0
                )
                > 1
            ):
                lightly_output = output_frame_sequence(
                    lightly_client, scheduled_run_id
                )
            else:
                lightly_output = output_sampled_images(lightly_client)
            logger.info(f"Lightly output {lightly_output}")
            return lightly_output

    with LightlyWorker(
        dataset_name=dataset_name,
        input_directory=input_directory,
        output_directory=output_directory,
        config=config,
        dataset_type=dataset_type,  # only videos have sequence info
        notify=notify,
        existing_dataset_id=dataset_id,
    ) as lightly_worker:
        return lightly_worker.run()
