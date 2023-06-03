#
# Copyright 2020-2022 Voxel Labs, Inc.
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
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import sematic
from loguru import logger
from sematic.resolvers.silent_resolver import SilentResolver
from tqdm import tqdm

from core.common.queries import FILTERED_RAW_INCIDENTS_QUERY
from core.metaverse.api.queries import is_data_collection_in_metaverse
from core.utils.aws_utils import (
    download_to_file,
    glob_from_bucket,
    upload_file,
)
from core.utils.perception_portal_graphql_session import (
    PerceptionPortalSession,
)
from core.utils.yaml_jinja import load_yaml_with_jinja


@dataclass
class IncidentFromPortalConfig:
    organizations: List[str]
    zones: List[str]
    cameras: List[str]
    incidents: Dict[str, str]


@dataclass
class Incident:
    camera_uuid: str
    video_s3_path: str
    incident_type_id: str
    incident_uuid: str
    organization: str
    zone: str
    experimental: bool
    feedback_type: str
    scenario_type: str
    copied_s3_path: Optional[str] = None
    created_at: Optional[str] = None


class DataFlywheelCollector:
    """Collects incidents from the portal for the data flywheel."""

    INCIDENTS_TO_FETCH_PER_QUERY = 20

    # trunk-ignore(pylint/R0913)
    def select_incidents_from_portal(
        self,
        config: List[IncidentFromPortalConfig],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_bucket: str = "voxel-lightly-input",
        output_path: str = "sandbox",
        max_num_incidents: int = 1000,
        allow_empty: bool = False,
        environment: str = "production",
        use_experimental_incidents: bool = False,
        metaverse_environment: Optional[str] = None,
    ) -> List[Incident]:
        """Queries incidents from portal and copies them to an S3 bucket

        Args:
            config (dict): dictionary with the above defined config format
            start_date (str, optional): The start date to query incidents from. Defaults to None.
            end_date (str, optional): The end date to query incidents to. Defaults to None.
            output_bucket (str, optional): S3 bucket name. Defaults to "voxel-lightly-input".
            output_path (str, optional): Path in S3 bucket. Defaults to "sandbox".
            max_num_incidents (int, optional): Max num of incidents to query. Defaults to 1000.
            allow_empty (bool, optional): Allows selecting zero incidents. Defaults to False.
            environment (str): Environment operation is being run in. Defaults to "production".
            use_experimental_incidents (bool): Flag to use experimental / nonexperimental incidents.
                Defaults to false
            metaverse_environment (Optional[str]): Metaverse environment to query datapool

        Returns:
            List[Incident]: list of incidents fetched

        Raises:
            RuntimeError: Throws if the query returns 0 incidents and allow_empty is False.
        """
        if start_date == "yesterday":
            start_date = (datetime.now() - timedelta(hours=24)).strftime(
                "%Y-%m-%d"
            )
        if start_date:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").isoformat()
        if end_date:
            end_date = datetime.strptime(end_date, "%Y-%m-%d").isoformat()

        incident_count = 0
        incident_list = []
        for config_item in config:
            logger.info(config_item)
            for incident_type, feedback_value in config_item.incidents.items():
                logger.info(f"{incident_type}, {feedback_value}")
                incidents = self.execute_query(
                    incident_type=incident_type,
                    feedback_type=feedback_value,
                    use_experimental_incidents=use_experimental_incidents,
                    organizations=config_item.organizations
                    if config_item.organizations
                    else None,
                    zones=config_item.zones if config_item.zones else None,
                    camera_uuids=config_item.cameras
                    if config_item.cameras
                    else None,
                    start_date=start_date,
                    end_date=end_date,
                    max_num_incidents=max_num_incidents,
                    metaverse_environment=metaverse_environment,
                )
                logger.info(f"Incidents: {incidents}")
                incident_count += len(incidents)
                if incidents:
                    self.copy_incidents_to_bucket(
                        incidents=incidents,
                        output_bucket=output_bucket,
                        output_path=output_path,
                        environment=environment,
                    )
                    incident_list.extend(incidents)
        if not allow_empty and incident_count == 0:
            raise RuntimeError(
                f"No incidents found with this config, {config}."
            )
        return incident_list

    # trunk-ignore(pylint/R0913)
    def execute_query(
        self,
        incident_type: str,
        feedback_type: str,
        use_experimental_incidents: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        organizations: Optional[List[str]] = None,
        zones: Optional[List[str]] = None,
        camera_uuids: Optional[List[str]] = None,
        max_num_incidents: Optional[int] = 1000,
        metaverse_environment: Optional[str] = None,
    ) -> List[Incident]:
        """Executes the portal query to select the desired incidents.

        Args:
            incident_type (str): type of incident
            feedback_type (str): type of feedback (valid, invalid, all)
            use_experimental_incidents(bool): flag to use experimental or nonexperimental incidents
            start_date (str, optional): date to select incidents from. Defaults to None.
            end_date (str, optional): date to copy incidents to. Defaults to None.
            organizations (list, optional): list of organizations to query. Defaults to None.
            zones (list, optional): list of zones to query. Defaults to None.
            camera_uuids (list, optional): list of cameras to query. Defaults to None.
            max_num_incidents (int, optional): max num of incidents to query. Defaults to 1000.
            metaverse_environment (Optional[str]): Metaverse environment to query datapool

        Raises:
            RuntimeError: Throws if the GraphQL query fails

        Returns:
            list: list of incidents
        """
        remaining_incidents_to_load = max_num_incidents
        incidents = []
        feedback_scenario_map = {
            "valid": "POSITIVE",
            "invalid": "NEGATIVE",
            "all": "UNKNOWN",
        }
        has_next_cursor = True
        end_cursor = None

        with PerceptionPortalSession("PROD") as perception_portal_session:
            while len(incidents) < max_num_incidents and has_next_cursor:
                scenarios_to_fetch = self.INCIDENTS_TO_FETCH_PER_QUERY

                scenarios_to_fetch = min(
                    remaining_incidents_to_load,
                    self.INCIDENTS_TO_FETCH_PER_QUERY,
                )

                # Create arguments dictionary
                variables = {
                    "fromUtc": start_date,
                    "toUtc": end_date,
                    "incidentTypeFilter": incident_type,
                    "feedbackType": feedback_type
                    if feedback_type != "all"
                    else None,
                    "first": scenarios_to_fetch,
                    "organizationKey": organizations
                    if organizations
                    else None,
                    "zoneKey": zones if zones else None,
                    "cameraUuid": camera_uuids if camera_uuids else None,
                    "after": end_cursor if end_cursor else None,
                }

                logger.info(f"Executing query with variables: {variables}")

                # Execute graphql query
                response = perception_portal_session.session.post(
                    f"{perception_portal_session.host}/graphql/",
                    json={
                        "query": FILTERED_RAW_INCIDENTS_QUERY,
                        "variables": variables,
                    },
                    headers=perception_portal_session.headers,
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Query failed: {response.reason}")

                result_json = json.loads(response.text)["data"][
                    "integrations"
                ]["filteredRawIncidents"]
                has_next_cursor = result_json["pageInfo"]["hasNextPage"]
                end_cursor = result_json["pageInfo"]["endCursor"]
                if remaining_incidents_to_load:
                    remaining_incidents_to_load = (
                        remaining_incidents_to_load - scenarios_to_fetch
                    )
                for node in result_json["edges"]:
                    # skip video if it is already in metaverse
                    if is_data_collection_in_metaverse(
                        node["node"]["uuid"],
                        metaverse_environment=metaverse_environment,
                    ):
                        remaining_incidents_to_load += 1
                        continue
                    # skip video if experimental flag does not match
                    if (
                        use_experimental_incidents
                        != node["node"]["experimental"]
                    ):
                        remaining_incidents_to_load += 1
                        continue

                    data = json.loads(node["node"]["data"])
                    incident = Incident(
                        camera_uuid=data["camera_uuid"],
                        video_s3_path=data["video_s3_path"],
                        incident_type_id=node["node"]["incidentType"]["key"],
                        incident_uuid=node["node"]["uuid"],
                        organization=node["node"]["organization"]["key"],
                        zone=node["node"]["zone"]["key"],
                        experimental=node["node"]["experimental"],
                        feedback_type=feedback_type,
                        scenario_type=feedback_scenario_map.get(
                            feedback_type, "UNKNOWN"
                        ),
                    )
                    incidents.append(incident)
        if len(incidents) > max_num_incidents:
            sorted_incidents = sorted(
                incidents, key=lambda incident: incident.incident_uuid
            )
            logger.warning(
                f"There were more incidents found: {len(incidents)} "
                f"in this time period than max: {max_num_incidents}, truncating"
            )
            return sorted_incidents[:max_num_incidents]
        return incidents

    def copy_incidents_to_bucket(
        self,
        incidents: List[Incident],
        output_bucket: str = "voxel_lightly_input",
        output_path: str = "sandbox",
        environment: str = "production",
    ):
        """Moves incident videos to a S3 bucket

        Args:
            incidents (list): list of incidents
            output_bucket (str, optional): Bucket to copy to. Defaults to "voxel_lightly_input".
            output_path (str, optional): Path in S3 bucket to copy videos to. Defaults to "sandbox".
            environment (str): Environment operation is being run in. Defaults to "production".

        """
        logger.info(f"Copying over {len(incidents)} incidents to AWS bucket..")
        for incident in tqdm(incidents):
            with tempfile.NamedTemporaryFile() as temporary_file:

                # Download incidents from aws
                incident_files_s3 = glob_from_bucket(
                    bucket=f"voxel-portal-{environment}",
                    # trunk-ignore(pylint/C0301): ok for line to be long since it is a prefix
                    prefix=f"{incident.organization.lower()}/{incident.zone.lower()}/incidents/{incident.incident_uuid}",
                    extensions=("mp4"),
                )

                def get_incident_videos(files: List[str]) -> List[str]:
                    """Helper method to get incident video

                    Args:
                        files (List[str]): list of files

                    Returns:
                        List[str]: list of files filtered
                    """

                    def filter_incident(file_type: str) -> List[str]:
                        """Helper method to filter incident files

                        Args:
                            file_type (str): file type to search for

                        Returns:
                            List[str]: filtered list of incidents
                        """
                        return [
                            incident_video_file
                            for incident_video_file in files
                            if file_type in incident_video_file
                        ]

                    incident_videos = filter_incident("original")  # h265 video
                    if not incident_videos:
                        incident_videos = filter_incident("mp4")  # h264 video

                    return incident_videos

                if len(incident_files_s3) > 0:

                    # download
                    incident_video = get_incident_videos(incident_files_s3)

                    download_to_file(
                        f"voxel-portal-{environment}",
                        incident_video[0],
                        temporary_file.name,
                    )

                    s3_path = upload_file(
                        bucket=output_bucket,
                        local_path=temporary_file.name,
                        s3_path=f"{output_path}/{incident.incident_uuid}.mp4",
                    )
                    incident.copied_s3_path = f"s3://{output_bucket}/{s3_path}"

        logger.info("Done")


def parse_args():
    """Parse CLI arguments

    Returns:
        dict: command line argument dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Config json path", required=True
    )
    parser.add_argument(
        "-s",
        "--start_date",
        type=str,
        default=None,
        help="Format 2021-10-25",
    )
    parser.add_argument(
        "-e",
        "--end_date",
        type=str,
        default=None,
        help="Format 2021-10-25",
    )
    parser.add_argument(
        "--output_bucket",
        type=str,
        default="voxel-lightly-input",
        help="S3 bucket to dump videos to",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="sandbox",
        help="Path of file in S3 bucket to dump videos to",
    )
    parser.add_argument(
        "--max_num_incidents",
        type=int,
        default=1000,
        help="Maximum number of incidents to download",
    )
    parser.add_argument(
        "--allow_empty",
        type=bool,
        default=False,
        help="Allows selecting zero incidents. Otherwise, raises an exception.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="production",
        help="Environment that operation should be running in",
    )
    parser.add_argument(
        "--use_experimental_incidents",
        action="store_true",
        help="Flag to use only experimental incidents compared to only non experimental incidents",
    )

    return parser.parse_known_args()[0]


@dataclass
class IncidentsFromPortalInput:
    """Input struct for incidents from portal"""

    config: List[IncidentFromPortalConfig]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    output_bucket: str = "voxel-lightly-input"
    output_path: str = "sandbox"
    max_num_incidents: int = 1000
    allow_empty: bool = False
    environment: str = "production"
    use_experimental_incidents: bool = False
    metaverse_environment: Optional[str] = None


@sematic.func
def select_incidents_from_portal(
    incidents_from_portal_input: IncidentsFromPortalInput,
) -> List[Incident]:
    """Sematified function for getting incidents from portal
    Args:
       incidents_from_portal_input (IncidentsFromPortalInput): incident input
    Returns:
        List[Incident]: list of incidents
    """
    flywheel_collector = DataFlywheelCollector()
    return flywheel_collector.select_incidents_from_portal(
        config=incidents_from_portal_input.config,
        start_date=incidents_from_portal_input.start_date,
        end_date=incidents_from_portal_input.end_date,
        output_bucket=incidents_from_portal_input.output_bucket,
        output_path=incidents_from_portal_input.output_path,
        max_num_incidents=incidents_from_portal_input.max_num_incidents,
        allow_empty=incidents_from_portal_input.allow_empty,
        environment=incidents_from_portal_input.environment,
        use_experimental_incidents=incidents_from_portal_input.use_experimental_incidents,
        metaverse_environment=incidents_from_portal_input.metaverse_environment,
    )


if __name__ == "__main__":
    args = parse_args()
    inp = IncidentsFromPortalInput(
        config=load_yaml_with_jinja(args.config),
        start_date=args.start_date,
        end_date=args.end_date,
        output_bucket=args.output_bucket,
        output_path=args.output_path,
        max_num_incidents=args.max_num_incidents,
        allow_empty=args.allow_empty,
        environment=args.environment,
        use_experimental_incidents=args.use_experimental_incidents,
        metaverse_environment=os.environ.get(
            "METAVERSE_ENVIRONMENT", "INTERNAL"
        ),
    )
    select_incidents_from_portal(inp).resolve(SilentResolver())
