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

import tempfile
from typing import List, Optional, Tuple

import sematic
import yaml
from loguru import logger

from core.infra.sematic.perception.data_ingestion.ingest_raw_videos.pipeline import (
    VideoIngestionSummary,
)
from core.infra.sematic.perception.data_ingestion.ingest_raw_videos.pipeline import (
    pipeline as raw_video_ingestion_pipeline,
)
from core.infra.sematic.shared.utils import PipelineSetup
from core.ml.data.collection.data_collector import (
    Incident,
    IncidentFromPortalConfig,
    IncidentsFromPortalInput,
    select_incidents_from_portal,
)
from core.utils.aws_utils import copy_object, upload_file
from core.utils.logging.list_indented_yaml_dumper import ListIndentedDumper


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def generate_scenario_config(
    incident_type: str, ingestion_summary: VideoIngestionSummary
) -> str:
    """Creates scenarios config from incidents and uploads to s3
    ## Parameters
    - **incident_type**
        Incident type for scenario set
    - **ingestion_summary**:
        Ingestion summary containing chunked video uuids

    ## Returns
        s3 path of scenarios config
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    run_id = sematic.context().run_id
    bucket = "voxel-temp"
    prefix = f"ingest_scenarios/{run_id}"
    file_name = f"scenario_set_{incident_type.lower()}.yaml"

    def create_config_from_incidents(
        incident_type: str, video_uuids: List[str]
    ) -> list:
        """Config creation helper
        Args:
            incident_type (str): incident type of scenario set
            video_uuids (List[str]): list of ingested video uuids
        Returns:
            scenario set config list
        """
        config = []
        for video_uuid in video_uuids:
            config.append(
                {
                    "camera_uuid": ("/").join(video_uuid.split("/")[0:4]),
                    "incidents": [incident_type.upper()]
                    if "POSITIVE" in video_uuid
                    else [],
                    "video_uuid": video_uuid,
                }
            )
        return config

    ingested_video_uuids = ingestion_summary.ingested_metaverse_videos
    positive_config = create_config_from_incidents(
        incident_type,
        [pos for pos in ingested_video_uuids if "POSITIVE" in pos],
    )
    positive_config_header = ("\n").join(
        (
            f"#### {incident_type}",
            f"# POSITIVE SCENARIOS - {len(positive_config)} videos\n",
        )
    )
    negative_config = create_config_from_incidents(
        incident_type,
        [neg for neg in ingested_video_uuids if "NEGATIVE" in neg],
    )
    negative_config_header = (
        f"# NEGATIVE SCENARIOS - {len(negative_config)} videos\n"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/{file_name}", "w", encoding="UTF-8") as file:
            file.write(positive_config_header)
            if len(positive_config) > 0:
                yaml.dump(positive_config, file, Dumper=ListIndentedDumper)
            file.write(negative_config_header)
            if len(negative_config) > 0:
                yaml.dump(negative_config, file, Dumper=ListIndentedDumper)
        upload_file(bucket, f"{temp_dir}/{file_name}", f"{prefix}/{file_name}")
    logger.info(
        f"Scenario config uploaded to s3://{bucket}/{prefix}/{file_name}"
    )
    return f"s3://{bucket}/{prefix}/{file_name}"


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def copy_incidents_to_voxel_raw_logs(incidents: List[Incident]) -> List[str]:
    """Move copied incidents to voxel-raw-logs for ingestion
    ## Parameters
    - **incidents**:
        Incident results from portal query

    ## Returns
        List of video uuids to ingest
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    bucket = "voxel-raw-logs"
    ext = "mp4"
    video_uuids = []
    for incident in incidents:
        video_uuid = (
            f"{incident.camera_uuid}/scenarios/"
            f"{incident.incident_type_id}/"
            f"{incident.scenario_type}/"
            f"{incident.incident_uuid}"
        )
        scenario_s3_uri = f"s3://{bucket}/{video_uuid}.{ext}"
        copy_object(incident.copied_s3_path, scenario_s3_uri)
        video_uuids.append(video_uuid)
    return video_uuids


# trunk-ignore-begin(pylint/W9015,pylint/W9011,pylint/R0913)
@sematic.func
def pipeline(
    incident_type: str,
    camera_uuids: Optional[List[str]],
    start_date: Optional[str],
    end_date: Optional[str],
    max_incidents: int,
    environment: str,
    experimental_incidents_only: bool,
    is_test: bool,
    fps: float,
    metaverse_environment: str,
    pipeline_setup: PipelineSetup,
) -> Tuple[VideoIngestionSummary, str]:
    """# Run scenario ingestion

    ## Parameters
    - **incident_type**:
        The incident type to query portal for
    - **camera_uuids**:
        Optional camera uuid parameter to limit portal query
    - **start_date**:
        Optional start date of incidents to limit portal query
    - **end_date**:
        Optional end date of incidents to limit portal query
    - **max_incidents**:
        Max number of incidents to query. Applies to both valid and invalid,
        so total number of incidents returned is 2 * max_incidents.
    - **environment**:
        Environment operation is being run in
    - **experimental_incidents_only**:
        Flag to restrict query to only get experimental incidents
    - **is_test**:
        Flag to use videos for testing
    - **fps**:
        FPS used to send videos for labeling. Anything below 0 results
        in videos not sent for labeling, 0 results in using original
        video FPS
    - **metaverse_environment**:
        Metaverse environment to use

    ## Returns
        A tuple containing the ingestion summary and the s3 path for the
        scenario config
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011,pylint/R0913)
    run_id = sematic.context().run_id
    scenarios_incident_config = [
        IncidentFromPortalConfig(
            organizations=[],
            zones=[],
            cameras=camera_uuids,
            incidents={
                incident_type.lower(): "valid",
            },
        ),
        IncidentFromPortalConfig(
            organizations=[],
            zones=[],
            cameras=camera_uuids,
            incidents={
                incident_type.lower(): "invalid",
            },
        ),
    ]
    data_collector_input = IncidentsFromPortalInput(
        config=scenarios_incident_config,
        start_date=start_date,
        end_date=end_date,
        output_bucket="voxel-temp",
        output_path=f"ingest_scenarios/{run_id}",
        max_num_incidents=max_incidents,
        environment=environment,
        use_experimental_incidents=experimental_incidents_only,
        metaverse_environment=metaverse_environment,
    )
    incidents = select_incidents_from_portal(data_collector_input)
    video_uuids_to_ingest = copy_incidents_to_voxel_raw_logs(incidents)
    ingestion_summary = raw_video_ingestion_pipeline(
        videos=video_uuids_to_ingest,
        is_test=is_test,
        fps=fps,
        metaverse_environment=metaverse_environment,
        pipeline_setup=PipelineSetup(),
    )
    scenario_path = generate_scenario_config(incident_type, ingestion_summary)
    return ingestion_summary, scenario_path
