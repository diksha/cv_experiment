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

# Usage: ./bazel run core/infra/sematic/perception/regression_scenarios:add_regression_scenario
# -- --incident_uuids <incident_uuids>

import argparse
import json
import os
from typing import List

from loguru import logger
from sematic import SilentResolver

from core.common.queries import INCIDENT_DETAILS
from core.infra.sematic.perception.data_ingestion.ingest_raw_videos.pipeline import (
    VideoIngestionSummary,
    pipeline,
)
from core.infra.sematic.shared.utils import PipelineSetup
from core.ml.data.collection.data_collector import (
    DataFlywheelCollector,
    Incident,
)
from core.utils.aws_utils import get_bucket_path_from_s3_uri
from core.utils.perception_portal_graphql_session import (
    PerceptionPortalSession,
)


def add_to_regression_set(
    ingested_videos_summary: VideoIngestionSummary, incidents: List[Incident]
):
    """
    Specify users what to add to regression set

    Args:
        ingested_videos_summary (VideoIngestionSummary): summary of ingested videos
        incidents (List[Incident]): list of incidents
    """
    incident_uuid_to_incident_map = {}
    for incident in incidents:
        incident_uuid_to_incident_map[incident.incident_uuid] = incident

    def get_incident_uuid(video_uuid) -> str:
        """
        Get incident uuid from video uuid

        Args:
            video_uuid (str): video uuid

        Returns:
            str: incident uuid

        Raises:
            RuntimeError: if video uuid is not valid
        """

        if video_uuid.endswith("_0000"):
            split_string = os.path.basename(video_uuid).rsplit("_0000", 1)
            return split_string[0]
        raise RuntimeError(f"{video_uuid} is not a valid video uuid")

    for (
        ingested_metaverse_video
    ) in ingested_videos_summary.ingested_metaverse_videos:
        incident_uuid = get_incident_uuid(ingested_metaverse_video)
        incident = incident_uuid_to_incident_map[incident_uuid]
        incidents_list = (
            incident.incident_type_id
            if incident.scenario_type == "POSTIVE"
            else ""
        )
        logger.info(
            (
                f"\nAdd following to {incident.incident_type_id.lower()}.yaml \n"
                f"  - camera_uuid: {incident.camera_uuid}\n"
                f"    incidents: [{incidents_list}]\n"
                f"    video_uuid: {ingested_metaverse_video}\n"
            )
        )


def ingest_incidents_to_metaverse(
    incidents: List[Incident], metaverse_environment: str
) -> VideoIngestionSummary:
    """Ingest incidents to metaverse

    Args:
        incidents (List[Incident]): list of incidents
        metaverse_environment (str): metaverse environment
    Returns:
        List[str]: list of video uuids
    """
    video_uuids = []
    for incident in incidents:
        video_uuids.append(
            os.path.splitext(
                get_bucket_path_from_s3_uri(incident.copied_s3_path)[1]
            )[0]
        )
    future = pipeline(
        videos=video_uuids,
        is_test=True,
        fps=-1,
        metaverse_environment=metaverse_environment,
        pipeline_setup=PipelineSetup(),
    )
    return future.resolve(SilentResolver())  # trunk-ignore(pylint/E1101)


def get_incidents_from_portal(incident_uuids: List[str]) -> List[Incident]:
    """
    Get incidents from perception portal

    Args:
        incident_uuids (List[str]): list of incident uuids

    Returns:
        List[Incident]: list of incidents
    """
    feedback_scenario_map = {
        "valid": "POSITIVE",
        "invalid": "NEGATIVE",
        "all": "UNKNOWN",
    }
    incidents = []
    with PerceptionPortalSession("PROD") as perception_portal_session:
        for incident_uuid in incident_uuids:
            variables = {"incidentUuid": incident_uuid}
            response = perception_portal_session.session.post(
                f"{perception_portal_session.host}/graphql/",
                json={
                    "query": INCIDENT_DETAILS,
                    "variables": variables,
                },
                headers=perception_portal_session.headers,
            )
            data = json.loads(
                json.loads(response.text)["data"]["incidentDetails"]["data"]
            )
            node = json.loads(response.text)["data"]["incidentDetails"]
            if node["validFeedbackCount"] > 0 >= node["invalidFeedbackCount"]:
                feedback = "valid"
            elif (
                node["invalidFeedbackCount"] > 0 >= node["validFeedbackCount"]
            ):
                feedback = "invalid"
            else:
                feedback = "unknown"

            incident = Incident(
                camera_uuid=data["camera_uuid"],
                video_s3_path=data["video_s3_path"],
                created_at=node["createdAt"],
                incident_type_id=data["incident_type_id"],
                incident_uuid=node["uuid"],
                organization=node["organization"]["key"],
                zone=node["zone"]["key"],
                experimental=node["experimental"],
                feedback_type=feedback,
                scenario_type=feedback_scenario_map.get(feedback, "UNKNOWN"),
            )
            incidents.append(incident)

    return incidents


def main(incident_uuids: List[str], metaverse_environment: str):
    logger.info(f"Fetching incidents from portal {incident_uuids}")
    incidents = get_incidents_from_portal(incident_uuids)
    logger.info("Copying incidents to voxel raw logs bucket")
    for incident in incidents:
        DataFlywheelCollector().copy_incidents_to_bucket(
            [incident],
            output_bucket="voxel-raw-logs",
            output_path=incident.camera_uuid,
        )
    logger.info(f"Ingesting incidents to metaverse {incidents}")
    ingested_videos = ingest_incidents_to_metaverse(
        incidents, metaverse_environment
    )
    logger.info(f"Videos ingestion summary {ingested_videos}")
    add_to_regression_set(ingested_videos, incidents)


def parse_args() -> argparse.Namespace:
    """
    Parse input commandline args

    Returns:
        argparse.Namespace: the parsed input commanline arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--incident_uuids",
        type=str,
        nargs="+",
        default=[],
        help="Incident uuids from portal",
    )
    parser.add_argument(
        "--metaverse_environment",
        type=str,
        default="INTERNAL",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.incident_uuids, args.metaverse_environment)
