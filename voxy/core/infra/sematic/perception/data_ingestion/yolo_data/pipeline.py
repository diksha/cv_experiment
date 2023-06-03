#
# Copyright 2022-2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import sematic
from loguru import logger

from core.infra.sematic.shared.resources import GPU_16CPU_64GB_1x
from core.infra.sematic.shared.utils import PipelineSetup
from core.labeling.logs_store.chunk_and_upload_video_logs import (
    chunk_and_upload_video_logs,
    create_video_ingest_input,
)
from core.labeling.logs_store.ingest_data_collection_metaverse import (
    ingest_data_collections_to_metaverse,
)
from core.labeling.scale.runners.create_scale_tasks import (
    ScaleTaskSummary,
    create_scale_tasks,
)
from core.labeling.tools.pull_kinesis_feed import PullFeedResult
from core.labeling.tools.pull_kinesis_feed_site import pull_kinesis_feed_site
from core.ml.data.curation.check_location_site import check_location_site
from core.ml.data.curation.lib.lightly_worker import run_lightly_worker
from core.ml.data.curation.prepare_lightly_run import (
    get_preparation_field,
    prepare_lightly_run,
)
from core.ml.data.curation.trim_lightly_clips import (
    get_output_bucket as get_trimmed_output_bucket,
)
from core.ml.data.curation.trim_lightly_clips import (
    get_to_ingest_videos,
    get_trimmed_video_uuids,
    trim_lightly_clips,
)
from core.utils.yaml_jinja import load_yaml_with_jinja

_LIGHTLY_CONFIG_PATH = (
    "core/ml/data/curation/configs/yolo_ingestion_lightly.yaml"
)


@dataclass
class IngestionSummary:
    """A summary of the data ingestion

    Attributes
    ----------
    camera_uuid:
        The UUID of the camera whose video was ingested
    ingestion_datetime:
        The date and time of the start of the video that was ingested
    scale_tasks:
        A summary of the tasks sent to Scale for labelling
    metaverse_ingested_uuids:
        A list of the UUIDs of the videos ingested to Metaverse
    metaverse_failed_ingest_uuids:
        A list of the UUIDs of the videos attempted to ingest to Metaverse which failed
    metaverse_environment:
        The metaverse environment videos were ingested to
    """

    camera_uuid: str
    ingestion_datetime: datetime
    scale_tasks: ScaleTaskSummary
    metaverse_ingested_uuids: List[str]
    metaverse_failed_ingest_uuids: List[str]
    metaverse_environment: str


@dataclass
class LightlyIngestionUserInput:
    """Class containing the attributes about camera batch info and lightly samples

    Attributes
    ----------
    lightly_num_samples:
        lightly_num_samples: For specific_camera_uuids, lightly num of samples camera subsets
    specific_camera_uuids:
        specific_camera_uuids: camera_uuids if we want to collect data for particular cameras

    """

    lightly_num_samples: int
    specific_camera_uuids: Optional[List[str]]


@sematic.func
def get_input_bucket(pull_kinesis_feed_results: List[PullFeedResult]) -> str:
    """Gets the input bucket for files from kinesis feed

    Args:
        pull_kinesis_feed_results (List[PullFeedResult]]): result from
        pull kinesis feed

    Raises:
        RuntimeError: No results in kinesis feed

    Returns:
        str: input bucket
    """
    failed_cameras = 0
    for pull_feed_result in pull_kinesis_feed_results:
        if not pull_feed_result.s3_path:
            logger.error(f"Failed for {pull_feed_result.camera_uuid}")
            failed_cameras += 1
    logger.info(f"Number of failed cameras is {failed_cameras}")
    if failed_cameras == len(pull_kinesis_feed_results):
        raise RuntimeError("All kinesis streams are down")
    for item in pull_kinesis_feed_results:
        if item.s3_path:
            return item.s3_path.replace("s3://", "").split("/")[0]
    raise RuntimeError("All kinesis streams are down")


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def ingest_object_detection_data(
    organization: str,
    location: str,
    ingestion_datetime: datetime,
    metaverse_environment: str,
    config: Dict[str, object],
    pipeline_setup: PipelineSetup,
    test_size: float,
    camera_batch_map: LightlyIngestionUserInput,
) -> Optional[IngestionSummary]:
    """# Ingest data from a camera, clean, subsample for labeling, and request labeling

    ## Parameters
    - **camera_uuid**:
        The UUID for the camera to pull the feed for.
    - **ingestion_datetime**:
        The time that the ingested video should start
    - **metaverse_environment**:
        The metaverse environment to ingest the resulting videos to
    - **max_videos**
        Maximum number of videos to pull per camera, 0 is unlimited
    - **config**
        The config to use to configure the ingestion object detection data
    - **test_size**:
        Size of test train split
    - **camera_batch_map**:
        class containing specific_camera_uuids list, lightly_num_samples

    ## Returns
    A summary of what was performed during the ingestion
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    run_id = sematic.context().run_id
    validate_cameras = check_location_site(
        organization, location, ingestion_datetime
    )
    if not validate_cameras:
        return None

    pull_kinesis_feed_results = pull_kinesis_feed_site(
        ingestion_datetime=ingestion_datetime,
        organization=organization,
        location=location,
        prefix=f"detector/{run_id}",
        specific_camera_uuids=camera_batch_map.specific_camera_uuids,
        **config.get("pull_kinesis_feed_site"),
    )
    camera_uuid = os.path.join(organization, location)
    input_bucket = get_input_bucket(pull_kinesis_feed_results)

    lightly_preparation_contents = prepare_lightly_run(
        input_bucket=input_bucket,
        camera_uuid=os.path.join(organization, location),
        project=f"detector/{run_id}",
        **config.get("prepare_lightly_run"),
    )

    lightly_config = load_yaml_with_jinja(_LIGHTLY_CONFIG_PATH)
    if camera_batch_map.lightly_num_samples > 0:
        lightly_config["selection_config"][
            "n_samples"
        ] = camera_batch_map.lightly_num_samples

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
        config=lightly_config,
    )
    lightly_sequence_specification.set(resource_requirements=GPU_16CPU_64GB_1x)

    trimmed_video_summary = trim_lightly_clips(
        input_bucket=input_bucket,
        camera_uuid=camera_uuid,
        sequence_information=lightly_sequence_specification,
        input_prefix=f"detector/{run_id}",
        **config.get("trim_lightly_clips"),
        test_size=test_size,
    )
    to_ingest_videos = get_to_ingest_videos(trimmed_video_summary)
    (
        data_collection_uuids,
        to_ingest_data_collections,
    ) = chunk_and_upload_video_logs(
        create_video_ingest_input(
            video_uuids=get_trimmed_video_uuids(trimmed_video_summary),
            input_bucket=get_trimmed_output_bucket(trimmed_video_summary),
            input_source="s3",
            metadata=to_ingest_videos,
        ),
    )
    scale_task_prefix = camera_uuid.replace("/", "_")
    scale_task_summary = create_scale_tasks(
        video_uuids=data_collection_uuids,
        prefix=scale_task_prefix,
        **config.get("create_scale_tasks"),
    )
    (
        metaverse_ingested_uuids,
        failed_ingest_uuids,
    ) = ingest_data_collections_to_metaverse(
        data_collections_metadata=to_ingest_data_collections,
        metaverse_environment=metaverse_environment,
    )

    return summarize(
        camera_uuid=camera_uuid,
        ingestion_datetime=ingestion_datetime,
        scale_tasks=scale_task_summary,
        metaverse_ingested_uuids=metaverse_ingested_uuids,
        metaverse_failed_ingest_uuids=failed_ingest_uuids,
        metaverse_environment=metaverse_environment,
    )


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def summarize(
    camera_uuid: str,
    ingestion_datetime: datetime,
    scale_tasks: ScaleTaskSummary,
    metaverse_ingested_uuids: List[str],
    metaverse_failed_ingest_uuids: List[str],
    metaverse_environment: str,
) -> IngestionSummary:
    """# Summarize the data ingestion

    This must be a Sematic function because the inputs to IngestionSummary
    are all futures when they are produced in the pipeline. Feeding them
    into a func before using them to instantiate the dataclass means they
    will be resolved.

    ## Parameters
    - **camera_uuid**:
        The UUID of the camera whose video was ingested
    - **ingestion_datetime**:
        The date and time of the start of the video that was ingested
    - **scale_tasks**:
        A summary of the tasks sent to Scale for labelling
    - **metaverse_ingested_uuids**:
        A list of the UUIDs of the videos ingested to Metaverse
    - **metaverse_failed_ingest_uuids**:
        A list of the UUIDs of the videos attemoted to ingest to Metaverse which failed
    - **metaverse_environment**:
        The metaverse environment being ingested to

    ## Returns
    A summary of what was performed during the ingestion
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    return IngestionSummary(
        camera_uuid=camera_uuid,
        ingestion_datetime=ingestion_datetime,
        scale_tasks=scale_tasks,
        metaverse_ingested_uuids=metaverse_ingested_uuids,
        metaverse_failed_ingest_uuids=metaverse_failed_ingest_uuids,
        metaverse_environment=metaverse_environment,
    )
