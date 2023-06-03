##
## Copyright 2020-2022 Voxel Labs, Inc.
## All rights reserved.
##
## This document may not be reproduced, republished, distributed, transmitted,
## displayed, broadcast or otherwise exploited in any manner without the express
## prior written permission of Voxel Labs, Inc. The receipt or possession of this
## document does not convey any rights to reproduce, disclose, or distribute its
## contents, or to manufacture, use, or sell anything that it may describe, in
## whole or in part.
##

"""
This is the module in which you define your pipeline functions.

Feel free to break these definitions into as many files as you want for your
preferred code structure.
"""

import os
import tempfile
from typing import List

import botocore.exceptions
import cv2
import numpy as np
import sematic
from loguru import logger

from core.infra.sematic.shared.resources import resources
from core.ml.data.collection.data_collector import (
    Incident,
    IncidentFromPortalConfig,
    IncidentsFromPortalInput,
    select_incidents_from_portal,
)
from core.ml.data.curation.lib.lightly_worker import run_lightly_worker
from core.ml.data.curation.prepare_lightly_run import (
    get_preparation_field,
    prepare_lightly_run,
)
from core.ml.data.curation.voxel_lightly_utils import LightlyVideoFrameSequence
from core.utils.aws_utils import (
    copy_object,
    download_to_file,
    glob_from_bucket,
    upload_cv2_imageobj_to_s3,
)
from core.utils.yaml_jinja import load_yaml_with_jinja

_PREPARE_LIGHTLY_RUN_DEFAULTS = {
    "output_bucket": "voxel-lightly-output",
    "postfix": "img/real/negative",
    "project": "spill_data",
}
_PORTAL_SEARCH_DEFAULTS = {
    "start_date": "2023-03-28",
    "end_date": "2023-04-13",
    "max_num_incidents": 15,
}

_TRAINING_OUTPUT_DEFAULTS = {
    "output_bucket": "voxel-users",
    "project": "common/spill_data",
}


@sematic.func
def pipeline(
    spill_config_path: str, lightly_config_path: str
) -> List[List[str]]:
    """Given a spill configs, run the data collector on camera_uuids in list
    Args:
        spill_config_path(str): A path to a yaml containing cameras,
                                sites and zones for data collection
        lightly_config_path(str): A path to a yaml containing lightly downsampling for spills
    Returns:
        List[List[str]]: Completion messages
    """
    logger.info("Starting pipeline")

    config = load_yaml_with_jinja(spill_config_path)
    camera_uuids = config[0].get("cameras")
    logger.info(f"Running data collection on {(camera_uuids)} cameras")
    sample_destination_paths = []
    for camera_uuid in camera_uuids:

        sample_destination_paths.append(
            run_data_collector(
                camera_uuid, spill_config_path, lightly_config_path
            )
        )
    return sample_destination_paths


def pull_from_portal(spill_config_path: str, camera_uuid: str) -> List:
    """Pull incidents from portal

    Args:
        spill_config_path (str): Config path for running scenarios from portal
        camera_uuid (str): Camera uuid to pull incidents from

    Returns:
        List: Incidents found within range specified in config
    """
    config = load_yaml_with_jinja(spill_config_path)
    config[0]["cameras"] = [camera_uuid]
    config = [
        IncidentFromPortalConfig(**config_item) for config_item in config
    ]
    incidents = select_incidents_from_portal(
        IncidentsFromPortalInput(
            config=config,
            output_bucket="voxel-lightly-input",
            output_path=os.path.join(
                _PREPARE_LIGHTLY_RUN_DEFAULTS["project"],
                camera_uuid,
                _PREPARE_LIGHTLY_RUN_DEFAULTS["postfix"],
            ),
            allow_empty=False,
            environment="production",
            metaverse_environment=os.environ.get(
                "METAVERSE_ENVIRONMENT", "INTERNAL"
            ),
            **_PORTAL_SEARCH_DEFAULTS,
        )
    )

    return incidents


@sematic.func
def lightly_downsampling(
    camera_uuid: str, lightly_config_path: str, incidents: List[Incident]
) -> List[LightlyVideoFrameSequence]:
    """Initialize and run lightly downsampling

    Args:
        camera_uuid (str): Camera uuid to downsample
        lightly_config_path (str): path to lightly downsampling config
        incidents (List[Incident]): list of incidents

    Returns:
        List[LightlyVideoFrameSequence]: frame sequences from lightly
    """
    # prepare for lightly downsampling
    lightly_preparation_contents = prepare_lightly_run(
        input_bucket="voxel-lightly-input",
        camera_uuid=camera_uuid,
        **_PREPARE_LIGHTLY_RUN_DEFAULTS,
    )
    # run lightly downsampling
    lightly_sequence_specification = run_lightly_worker(
        dataset_id=get_preparation_field(
            lightly_preparation_contents, "dataset_id"
        ),
        dataset_name=get_preparation_field(
            lightly_preparation_contents, "dataset_name"
        ),
        input_directory=get_preparation_field(
            lightly_preparation_contents, "input_dir"
        ),
        output_directory=get_preparation_field(
            lightly_preparation_contents, "output_dir"
        ),
        config=load_yaml_with_jinja(lightly_config_path),
    )
    return lightly_sequence_specification


@sematic.func(
    resource_requirements=resources(cpu="900m", memory="3800M"),
    standalone=True,
)
def annotate_copy_files(
    camera_uuid: str,
    lightly_sequence_specification: List[LightlyVideoFrameSequence],
) -> List[str]:
    """Generate annotations for images and copy to destination s3 location

    Args:
        camera_uuid (str): camera_uuid for current run
        lightly_sequence_specification (List[LightlyVideoFrameSequence]): list of lightly jobs
    Returns:
        List[str]: sample destination path
    """

    img_filenames = glob_from_bucket(
        bucket=_PREPARE_LIGHTLY_RUN_DEFAULTS["output_bucket"],
        prefix=os.path.join(
            _PREPARE_LIGHTLY_RUN_DEFAULTS["project"],
            camera_uuid,
            f'{_PREPARE_LIGHTLY_RUN_DEFAULTS["postfix"]}/.lightly/frames',
        ),
        extensions=(".png"),
    )

    for img_filename in img_filenames:
        with tempfile.NamedTemporaryFile(
            suffix=".png"
        ) as img, tempfile.NamedTemporaryFile(suffix=".png") as annotation:
            download_to_file(
                _PREPARE_LIGHTLY_RUN_DEFAULTS["output_bucket"],
                img_filename,
                img.name,
            )
            image = cv2.imread(img.name)
            annotation = np.zeros_like(image, dtype=int)
            dest_img_filename = (
                img_filename.replace(
                    f'{_PREPARE_LIGHTLY_RUN_DEFAULTS["project"]}/', ""
                )
                .replace("/cha/", "/")
                .replace("/.lightly/frames/", "/")
            )
            # copy
            copy_object(
                source_uri=os.path.join(
                    "s3://",
                    _PREPARE_LIGHTLY_RUN_DEFAULTS["output_bucket"],
                    img_filename,
                ),
                destination_uri=os.path.join(
                    "s3://",
                    _TRAINING_OUTPUT_DEFAULTS["output_bucket"],
                    _TRAINING_OUTPUT_DEFAULTS["project"],
                    dest_img_filename,
                ),
            )
            # write annotation
            annotation_filename = dest_img_filename.replace(
                "/img/", "/annotation/"
            )
            logger.info(f"Destination filepath: {dest_img_filename}")
            upload_cv2_imageobj_to_s3(
                path=os.path.join(
                    "s3://",
                    _TRAINING_OUTPUT_DEFAULTS["output_bucket"],
                    _TRAINING_OUTPUT_DEFAULTS["project"],
                    annotation_filename,
                ),
                image=annotation,
            )
    return [
        os.path.join(
            "s3://",
            _TRAINING_OUTPUT_DEFAULTS["output_bucket"],
            _TRAINING_OUTPUT_DEFAULTS["project"],
            dest_img_filename,
        )
    ]


@sematic.func(
    resource_requirements=resources(cpu="900m", memory="3800M"),
    standalone=True,
    retry=sematic.RetrySettings(
        exceptions=(botocore.exceptions.ClientError,), retries=4
    ),
)
def run_data_collector(
    camera_uuid: str,
    spill_config_path: str,
    lightly_config_path: str,
) -> List[str]:
    """Runs the scenario set with given model and machine parameters
    Args:
        camera_uuid (str): camera_uuid to run data collector
        spill_config_path (str): path to spill config
        lightly_config_path (str): path to lightly downsampling config
    Returns:
        List[str]: Sample destination path
    """

    logger.info(f"FP data collection from camera {camera_uuid}")
    os.environ["METAVERSE_ENVIRONMENT"] = "INTERNAL"
    try:
        incidents = pull_from_portal(spill_config_path, camera_uuid)
    except RuntimeError:
        logger.debug(f"No incidents for camera {camera_uuid}")
        return [""]

    logger.info(
        f"Completed portal FP spill data collection from camera {camera_uuid}"
    )

    lightly_sequence_specification = lightly_downsampling(
        camera_uuid, lightly_config_path, incidents
    )

    logger.info(f"Completed lightly downsampling from camera {camera_uuid}")

    destination_filepaths = annotate_copy_files(
        lightly_sequence_specification=lightly_sequence_specification,
        camera_uuid=camera_uuid,
    )

    logger.info(f"Completed annotation creation for {camera_uuid}")

    return destination_filepaths
