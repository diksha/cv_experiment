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
import typing
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import sematic

from core.infra.sematic.shared.aws_funcs import get_bucket_from_s3_uri
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
from core.labeling.tools.pull_kinesis_feed import PullFeedResult, pull_feed
from core.ml.data.curation.check_camera_uuid import (
    PipelineIngestException,
    camera_config_valid,
    date_after_retention_period,
)
from core.ml.data.curation.crop_s3_videos import crop_all_doors_from_videos
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
    "core/ml/data/curation/configs/DOOR_STATE_DATA_INGESTION.yaml"
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


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def get_bucket_from_results(full_path: PullFeedResult) -> str:
    """# Given pull feed result, get the bucket

    ## Parameters
    - **full_path**:
        An s3 URI like s3://my-bucket/my/path

    ## Returns
    The bucket name
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    return full_path.s3_path.replace("s3://", "").split("/")[0]


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def get_path_from_results(full_path: PullFeedResult) -> str:
    """# Given an S3 URI, get the path

    ## Parameters
    - **full_path**:
        An s3 URI like s3://my-bucket/my/path

    ## Returns
    The path portion of the URI that follows the bucket name
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    return "/".join(full_path.s3_path.replace("s3://", "").split("/")[1:])


# trunk-ignore-begin(pylint/W9015,pylint/W9011,pylint/W9006)
@sematic.func
def validate_pull_feed_results(
    camera_s3_path: PullFeedResult,
) -> PullFeedResult:
    """# Validates the result of the kinesis stream pull

    ## Parameters
    - **camera_s3_path**:
        A pull feed result from `pull_feed`

    ## Raises
       RuntimeError if the feed pull was unsuccessful

    ## Returns
    The pull feed result if it is valid
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011,pylint/W9006)
    if camera_s3_path.s3_path is None:
        raise RuntimeError(
            "The feed was not available on this date,"
            " please try again on another day"
        )
    return camera_s3_path


# trunk-ignore-begin(pylint/W9015,pylint/W9011,pylint/W9006)
@sematic.func
def ingest_door_data(
    camera_uuid: str,
    ingestion_datetime: datetime,
    metaverse_environment: str,
    max_videos: int,
    test_size: float,
    config: typing.Dict[str, object],
    pipeline_setup: PipelineSetup,
    kinesis_upload_s3_path: Optional[str] = None,
) -> IngestionSummary:
    """# Ingest data from a camera, clean, subsample for labeling, and request labeling

    ## Parameters
    - **camera_uuid**:
        The UUID for the camera to pull the feed for.
    - **ingestion_datetime**:
        The time that the ingested video should start
    - **metaverse_environment**:
        The metaverse environment to ingest the resulting videos to
    - **max_videos**:
        Maximum number of videos to upload
    - **test_size**:
        Size of test train split
    - **kinesis_upload_s3_path**:
        To use data that was *already pulled* from Kinesis, this can be specified to be
        an S3 path containing the data from Kinesis. If left as None, data will be
        pulled from Kinesis.
    - **config**:
       Config to run pipeline end to end

    ## Returns
    A summary of what was performed during the ingestion
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011,pylint/W9006)
    run_id = sematic.context().run_id
    if not camera_config_valid(camera_uuid):
        raise PipelineIngestException(
            f"Invalid camera uuid: '{camera_uuid}'. Check logs for reason."
        )
    if not date_after_retention_period(ingestion_datetime):
        raise PipelineIngestException(
            f"Date: {ingestion_datetime} is before the retention period"
        )
    if kinesis_upload_s3_path is None:
        pull_feed_results = pull_feed(
            ingestion_datetime=ingestion_datetime,
            camera_uuid=camera_uuid,
            prefix=f"doors/{run_id}",
            max_videos=max_videos,
            **config.get("pull_feed"),
        )
    pull_feed_results = validate_pull_feed_results(pull_feed_results)
    cropped_videos_s3_path = crop_all_doors_from_videos(
        input_bucket=get_bucket_from_results(pull_feed_results),
        video_path=get_path_from_results(pull_feed_results),
        camera_uuid=camera_uuid,
        project=f"doors/cropped/{run_id}",
        input_prefix_to_remove=f"doors/{run_id}",
        **config.get("crop_videos"),
    )
    cropped_videos_bucket = get_bucket_from_s3_uri(cropped_videos_s3_path)
    lightly_preparation_contents = prepare_lightly_run(
        input_bucket=cropped_videos_bucket,
        camera_uuid=camera_uuid,
        project=f"doors/cropped/{run_id}",
        **config.get("prepare_lightly_run"),
    )

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
        config=load_yaml_with_jinja(_LIGHTLY_CONFIG_PATH),
    )
    trimmed_video_summary = trim_lightly_clips(
        input_bucket=cropped_videos_bucket,
        camera_uuid=camera_uuid,
        sequence_information=lightly_sequence_specification,
        input_prefix=f"doors/{run_id}",
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
        **config.get("create_scale_task"),
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
