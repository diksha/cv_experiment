"""Daemon which publishes incidents downstream as they become available.

Run with:
    ./bin/python core/incidients/publisher.py
"""
import argparse
import glob
import json
import logging
import os
import time
import traceback
import typing as t
from pathlib import Path

import requests

from core.structs.incident import Incident
from core.utils.aws_utils import upload_file
from services.perception.incidents.aggregation.controller import (
    IncidentAggregationController,
)

logging.basicConfig(level=logging.INFO)


class Publisher:
    def __init__(
        self,
        portal_host: str = None,
        auth_token: str = None,
        organization_key: str = None,
        temp_directory: str = "",
        should_generate_cooldown_incidents=False,
        **kwargs,
    ):
        del kwargs
        self._portal_host = portal_host
        self._auth_token = auth_token
        self._organization_key = organization_key
        self._temp_directory = temp_directory
        self._incident_json_glob = os.path.join(
            temp_directory, "*_incident.json"
        )

        self._s3_bucket = get_s3_bucket()
        self._s3_bucket_mumbai = get_s3_bucket_mumbai()

        self._finalized = False

        # Ensure host directory exists and init shared session
        Path(self._temp_directory).mkdir(parents=True, exist_ok=True)
        with requests.Session() as session:
            self._session = session
        self._should_generate_cooldown_incidents = (
            should_generate_cooldown_incidents
        )
        self._incident_aggregation_controller = IncidentAggregationController()

    def run(self):
        # TODO: log/alert on exceptions/gracefully degrade (move broken assets
        # out of the way)
        logging.info("Starting publisher daemon...")
        while not self._finalized:
            incident_json_filepaths = self.get_incident_json_filepaths(
                self._incident_json_glob
            )
            for filepath in incident_json_filepaths:
                self._publish_incident(filepath)
            time.sleep(1)

    def finalize(self):
        while len(glob.glob(self._incident_json_glob)):
            logging.debug("Publisher Finalizing")
            time.sleep(10)
            self._finalized = True

    def get_incident_json_filepaths(self, glob_pattern: str) -> t.List[str]:
        """Get incident JSON filepaths, sorted by mtime ascending.

        Args:
            glob_pattern (str): glob pattern

        Returns:
            t.List[str]: sorted list of filepaths
        """
        return sorted(glob.glob(glob_pattern), key=os.path.getmtime)

    def _publish_incident(self, incident_filepath):
        # TODO: validate all necessary data is present
        with open(incident_filepath, "r") as f:
            incident = Incident.from_dict(json.load(f))
        incident.organization_key = self._organization_key

        org_key = incident.organization_key.lower()
        site_key = incident.camera_uuid.split("/")[1].lower()

        if (
            not incident.cooldown_tag
            or self._should_generate_cooldown_incidents
        ):
            try:
                incident.video_thumbnail_s3_path = self._upload_thumbnail(
                    org_key=org_key,
                    site_key=site_key,
                    incident_uuid=incident.uuid,
                )
                incident.video_s3_path = self._upload_video(
                    org_key=org_key,
                    site_key=site_key,
                    incident_uuid=incident.uuid,
                )
                incident.original_video_s3_path = self._upload_original_video(
                    org_key=org_key,
                    site_key=site_key,
                    incident_uuid=incident.uuid,
                )
                incident.annotations_s3_path = self._upload_annotations(
                    org_key=org_key,
                    site_key=site_key,
                    incident_uuid=incident.uuid,
                )

                # TODO: attach log path to incident model in portal
                self._upload_log(
                    org_key=org_key,
                    site_key=site_key,
                    incident_uuid=incident.uuid,
                )
                self._upload_incident(
                    org_key=org_key,
                    site_key=site_key,
                    incident_uuid=incident.uuid,
                    incident_filepath=incident_filepath,
                )
            # trunk-ignore(pylint/W0703): operation needs to continue if any errors thrown
            except Exception:
                logging.warning(traceback.format_exc())

            # Run Incident aggregation before publishing to portal
            # if incident can be aggregated then a head incident will be returned
            # with updated fields, otherwise the original incident will be returned
            # and it may be cached in the aggregator for future aggregation.
            aggregated_incident = (
                self._incident_aggregation_controller.process(incident)
            )
            headers = {"Authorization": f"Token {self._auth_token}"}
            response = self._session.post(
                f"{self._portal_host}/api/incidents/",
                headers=headers,
                data=aggregated_incident.to_dict(),
            )
            logging.info(  # trunk-ignore(pylint/W1203)
                f"Response from portal: {response.text}"
            )

        # TODO: move to "processed" directory if we want to keep a copy?
        incident_glob = os.path.join(self._temp_directory, f"{incident.uuid}*")
        for filepath in glob.glob(incident_glob):
            Path(filepath).unlink()

    def _upload_thumbnail(self, org_key, site_key, incident_uuid):
        """Helper method to upload thumbnail jpg to s3

        Args:
            org_key (str): key of the org the incident occured in
            site_key (str): key of the site the incident occurred in
            incident_uuid (str): uuid of incident

        Returns:
            str: string value of s3 path that is uploaded to
        """
        local_path = os.path.join(
            self._temp_directory, f"{incident_uuid}_thumbnail.jpg"
        )

        s3_path = (
            f"{org_key}/{site_key}/incidents/{incident_uuid}_thumbnail.jpg"
        )
        upload_file(
            self._s3_bucket,
            local_path,
            s3_path,
            extra_args={
                "ContentType": "image/jpeg",
            },
        )
        return f"s3://{self._s3_bucket}/{s3_path}"

    def _upload_video(self, org_key, site_key, incident_uuid):
        """Helper method to upload video to s3

        Args:
            org_key (str): key of the org the incident occured in
            site_key (str): key of the site the incident occurred in
            incident_uuid (str): uuid of incident

        Returns:
            str: string value of s3 path that file is uploaded to
        """
        final_path = os.path.join(
            self._temp_directory, f"{incident_uuid}_video.mp4"
        )

        s3_path = f"{org_key}/{site_key}/incidents/{incident_uuid}_video.mp4"
        for s3_bucket in set([self._s3_bucket, self._s3_bucket_mumbai]):
            upload_file(
                s3_bucket,
                final_path,
                s3_path,
                extra_args={
                    "ContentType": "video/mp4",
                },
            )

        return f"s3://{self._s3_bucket}/{s3_path}"

    def _upload_original_video(self, org_key, site_key, incident_uuid):
        """Helper method to upload original video to s3

        Args:
            org_key (str): key of the org the incident occured in
            site_key (str): key of the site the incident occurred in
            incident_uuid (str): uuid of incident

        Returns:
            str: string value of s3 path that file is uploaded to
        """
        logging.info("Uploading original video for %s", incident_uuid)
        files = glob.glob(
            os.path.join(
                self._temp_directory, f"{incident_uuid}_original_video*"
            )
        )
        if len(files) != 1:
            logging.warning(
                "Needs to be 1 file for %s. Found %d files",
                incident_uuid,
                len(files),
            )
            return ""
        final_path = files[0]
        s3_path = (
            f"{org_key}/{site_key}/incidents/{os.path.basename(final_path)}"
        )
        logging.info("Uploading file to %s", s3_path)

        upload_file(
            self._s3_bucket,
            final_path,
            s3_path,
            extra_args={
                "ContentType": "video/mp4",
            },
        )

        return f"s3://{self._s3_bucket}/{s3_path}"

    def _upload_log(self, org_key, site_key, incident_uuid):
        """Helper method to upload mcap log to s3

        Args:
            org_key (str): key of the org the incident occured in
            site_key (str): key of the site the incident occurred in
            incident_uuid (str): uuid of incident

        Returns:
            str: string value of s3 path that file is uploaded to
        """
        logging.info("Uploading incident log for %s", incident_uuid)
        files = glob.glob(
            os.path.join(self._temp_directory, f"{incident_uuid}_log*")
        )
        if len(files) != 1:
            logging.warning(
                "Needs to be 1 file for %s. Found %d files",
                incident_uuid,
                len(files),
            )
            return ""
        final_path = files[0]
        s3_path = f"{org_key}/{site_key}/logs/{os.path.basename(final_path)}"
        logging.info("Uploading file to %s", s3_path)

        upload_file(
            self._s3_bucket,
            final_path,
            s3_path,
        )

        return f"s3://{self._s3_bucket}/{s3_path}"

    def _upload_annotations(self, org_key, site_key, incident_uuid):
        """Helper method to upload annotations json to s3

        Args:
            org_key (str): key of the org the incident occured in
            site_key (str): key of the site the incident occurred in
            incident_uuid (str): uuid of incident

        Returns:
            str: string value of s3 path that file is uploaded to
        """

        local_path = os.path.join(
            self._temp_directory, f"{incident_uuid}_annotations.json"
        )

        s3_path = (
            f"{org_key}/{site_key}/incidents/{incident_uuid}_annotations.json"
        )

        for s3_bucket in set([self._s3_bucket, self._s3_bucket_mumbai]):
            upload_file(
                s3_bucket,
                local_path,
                s3_path,
                extra_args={
                    "ContentType": "application/json",
                },
            )

        return f"s3://{self._s3_bucket}/{s3_path}"

    def _upload_incident(
        self, org_key, site_key, incident_uuid, incident_filepath
    ):
        """Helper method to upload incident json to s3

        Args:
            org_key (str): key of the org the incident occured in
            site_key (str): key of the site the incident occurred in
            incident_uuid (str): uuid of incident
            incident_filepath (str): path to incident json file

        Returns:
            str: string value of s3 path that file is uploaded to
        """
        s3_path = f"{org_key}/{site_key}/incidents/{incident_uuid}.json"
        upload_file(
            self._s3_bucket,
            incident_filepath,
            s3_path,
            extra_args={
                "ContentType": "application/json",
            },
        )

        return f"s3://{self._s3_bucket}/{s3_path}"


def get_arguments():
    parser = argparse.ArgumentParser(description="Incident Publisher")
    parser.add_argument(
        "--portal_host", type=str, help="Hostname of the portal backend."
    )
    parser.add_argument(
        "--organization_key",
        type=str,
        help="Organization key used to map incidents to organizations.",
    )
    parser.add_argument(
        "--auth_token",
        type=str,
        help="Portal auth token, tied to a single organization.",
    )
    return parser.parse_args()


def get_portal_host(args):
    return (
        args.portal_host
        or os.getenv("VOXEL_PORTAL_HOST")
        or "https://app.voxelai.com"
    )


def get_organization_key(args):
    return (
        args.organization_key
        or os.getenv("VOXEL_PORTAL_ORGANIZATION_KEY")
        # TODO: deprecate this value, auth_tokens should be associated with a
        # single organization
        or "VOXEL_SANDBOX"
    )


def get_portal_auth_token(args):
    return (
        args.portal_host
        or os.getenv("VOXEL_PORTAL_AUTH_TOKEN")
        # Default sandbox token for development/testing
        or "6bdeb3310546f9cae57c87604f80ece37ba6da43"
    )


def get_s3_bucket():
    """Get s3 bucket path depending on environment variable

    Returns:
        str: return bucket name
    """
    return (
        (
            os.getenv("ENVIRONMENT") == "production"
            and "voxel-portal-production"
        )
        or (os.getenv("ENVIRONMENT") == "staging" and "voxel-portal-staging")
        or "voxel-development-perception-incident-media"
    )


def get_s3_bucket_mumbai():
    """Get Mumbai region s3 bucket path depending on environment variable

    Returns:
        str: return bucket name
    """
    return (
        (
            os.getenv("ENVIRONMENT") == "production"
            and "voxel-portal-production-mumbai"
        )
        or (
            os.getenv("ENVIRONMENT") == "staging"
            and "voxel-portal-staging-mumbai"
        )
        or "voxel-development-perception-incident-media"
    )


if __name__ == "__main__":
    args = get_arguments()
    portal_host = get_portal_host(args)
    auth_token = get_portal_auth_token(args)
    organization_key = get_organization_key(args)
    publisher = Publisher(
        portal_host=portal_host,
        auth_token=auth_token,
        organization_key=organization_key,
    )
    publisher.run()
