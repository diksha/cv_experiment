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
from dataclasses import dataclass
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
from core.metaverse.api.queries import get_or_create_task_and_service
from core.ml.common.utils import get_merged_config
from core.ml.data.collection.data_collector import (
    Incident,
    IncidentsFromPortalInput,
    select_incidents_from_portal,
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
from core.structs.model import ModelCategory
from core.structs.task import TaskPurpose
from core.utils.yaml_jinja import load_yaml_with_jinja

_CROP_VIDEOS_DEFAULTS = dict(
    output_bucket="voxel-lightly-input",
    extension="mp4",
)

_PREPARE_LIGHTLY_RUN_DEFAULTS = dict(
    output_bucket="voxel-lightly-output",
)

_TRIM_LIGHTLY_CLIPS_DEFAULTS = dict(
    output_bucket="voxel-lightly-output",
    task="door",
    remove_task_name=False,
    use_epoch_time=False,
)

_DATA_DATAFLYWHEEL_CONFIG_PATH = (
    "core/infra/sematic/perception/data_ingestion/"
    "dataflywheel/configs/DOOR_STATE.yaml"
)


@dataclass
class IngestionSummary:
    """A summary of the data ingestion

    Attributes
    ----------
    camera_uuid:
        The UUID of the camera whose video was ingested
    start_date:
        The start date for the query used for incidents
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
    start_date: Optional[str]
    scale_tasks: ScaleTaskSummary
    metaverse_ingested_uuids: List[str]
    metaverse_failed_ingest_uuids: List[str]
    metaverse_environment: str


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def get_bucket_from_incidents(incidents: List[Incident]) -> str:
    """# Given pull feed result, get the bucket

    ## Parameters
    - **incidents**:
        list of incidents from the datacollector

    ## Returns
    The bucket name
    """
    copied_s3_path = incidents[0].copied_s3_path
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    return copied_s3_path.replace("s3://", "").split("/")[0]


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def door_dataflywheel(
    camera_uuid: str,
    start_date: Optional[str],
    max_incidents: int,
    metaverse_environment: str,
    overwrite_config_file: Optional[str],
    pipeline_setup: PipelineSetup,
) -> IngestionSummary:
    """# Run Dataflywheel For Doors

    Note: This is a temporary pipeline, while work is done to
    integrate data preprocessing before lightly step in regular
    dataflywheel paradigm.

    ## Parameters
    - **camera_uuid**:
        The UUID for the camera to pull the feed for.
    - **start_date**:
        The date that the dataflywheel query should start
    - **metaverse_environment**:
        The metaverse environment to ingest the resulting videos to
    - **kinesis_upload_s3_path**:
        To use data that was *already pulled* from Kinesis, this can be specified to be
        an S3 path containing the data from Kinesis. If left as None, data will be
        pulled from Kinesis.
    - **overwrite_config_file**:
        (Optional[str]): config that will overwrite dataflywheel configs


    ## Returns
    A summary of what was performed during the ingestion
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    run_id = sematic.context().run_id
    dataflywheel_config = load_yaml_with_jinja(
        _DATA_DATAFLYWHEEL_CONFIG_PATH,
    )
    overwrite_config = (
        load_yaml_with_jinja(overwrite_config_file)
        if overwrite_config_file
        else {}
    )
    task = get_or_create_task_and_service(
        TaskPurpose["DOOR_STATE"],
        ModelCategory["IMAGE_CLASSIFICATION"],
        camera_uuid,
        metaverse_environment=metaverse_environment,
    )
    incidents = select_incidents_from_portal(
        IncidentsFromPortalInput(
            config=get_merged_config(
                load_yaml_with_jinja(
                    dataflywheel_config["data_collection_config_file"],
                    task=task.to_dict(),
                ),
                overwrite_config.get("data_collection_config"),
            ),
            start_date=start_date,
            output_path=f"doors/{run_id}/{camera_uuid}",
            max_num_incidents=max_incidents,
            metaverse_environment=metaverse_environment,
        )
    )
    cropped_videos_s3_path = crop_all_doors_from_videos(
        input_bucket=get_bucket_from_incidents(incidents),
        video_path=f"doors/{run_id}/{camera_uuid}",
        camera_uuid=camera_uuid,
        project=f"doors/cropped/{run_id}",
        input_prefix_to_remove=f"doors/{run_id}",
        **_CROP_VIDEOS_DEFAULTS,
    )
    cropped_videos_bucket = get_bucket_from_s3_uri(cropped_videos_s3_path)
    lightly_preparation_contents = prepare_lightly_run(
        input_bucket=cropped_videos_bucket,
        camera_uuid=camera_uuid,
        project=f"doors/cropped/{run_id}",
        **_PREPARE_LIGHTLY_RUN_DEFAULTS,
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
        config=get_merged_config(
            load_yaml_with_jinja(dataflywheel_config["lightly_config_file"]),
            overwrite_config.get("lightly_config"),
        ),
    )
    trimmed_video_summary = trim_lightly_clips(
        input_bucket=cropped_videos_bucket,
        camera_uuid=camera_uuid,
        sequence_information=lightly_sequence_specification,
        input_prefix=f"doors/{run_id}",
        **_TRIM_LIGHTLY_CLIPS_DEFAULTS,
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
    ingestion_config = get_merged_config(
        load_yaml_with_jinja(dataflywheel_config["ingestion_config_file"]),
        overwrite_config.get("ingestion_config", None),
    )
    scale_task_prefix = camera_uuid.replace("/", "_")
    scale_task_summary = create_scale_tasks(
        video_uuids=data_collection_uuids,
        prefix=scale_task_prefix,
        **ingestion_config,
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
        start_date=start_date,
        scale_tasks=scale_task_summary,
        metaverse_ingested_uuids=metaverse_ingested_uuids,
        metaverse_failed_ingest_uuids=failed_ingest_uuids,
        metaverse_environment=metaverse_environment,
    )


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def summarize(
    camera_uuid: str,
    start_date: Optional[str],
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
    - **start_date**:
        The start date for the query used for incidents
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
        start_date=start_date,
        scale_tasks=scale_tasks,
        metaverse_ingested_uuids=metaverse_ingested_uuids,
        metaverse_failed_ingest_uuids=metaverse_failed_ingest_uuids,
        metaverse_environment=metaverse_environment,
    )
