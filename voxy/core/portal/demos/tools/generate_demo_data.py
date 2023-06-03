# trunk-ignore-all(pylint/C0413,flake8/E402): must setup django first before some imports
# trunk-ignore-all(bandit/B311): do not need crytographic randomness here

import argparse
import os
import random
import typing as t
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from uuid import uuid4

import django
from loguru import logger

from core.utils.aws_utils import (
    copy_object,
    separate_bucket_from_relative_path,
)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.portal.voxel.settings")
django.setup()

from core.portal.api.models.incident import Incident
from core.portal.api.models.incident_type import (
    CameraIncidentType,
    IncidentType,
    OrganizationIncidentType,
    SiteIncidentType,
)
from core.portal.api.models.organization import Organization
from core.portal.demos.data.constants import (
    DEMO_CAMERA_UUID_PREFIX,
    DEMO_ORG_KEY,
    DEMO_SITE_KEY,
)
from core.portal.demos.data.incident_types import INCIDENT_TYPE_CONFIGS
from core.portal.demos.data.types import DemoIncidentTypeConfig
from core.portal.devices.models.camera import Camera
from core.portal.incidents.enums import ReviewStatus
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.zones.models.zone import Zone


@dataclass
class Arguments:
    incident_type_key: str


@dataclass
class DemoDataGenerator:
    @cached_property
    def _organization(self) -> Organization:
        return Organization.objects.get(key=DEMO_ORG_KEY)

    @cached_property
    def _site(self) -> Zone:
        return Zone.objects.get(
            organization__key=DEMO_ORG_KEY, key=DEMO_SITE_KEY
        )

    @cached_property
    def _cameras(self) -> t.List[Camera]:
        return list(self._site.cameras.all())

    @cached_property
    def _incident_type_map(self) -> t.Dict[str, IncidentType]:
        return {item.key.upper(): item for item in IncidentType.objects.all()}

    def _get_config(self, incident_type_key: str) -> DemoIncidentTypeConfig:
        if incident_type_key not in INCIDENT_TYPE_CONFIGS:
            raise RuntimeError(
                f"No incident type config for key: {incident_type_key}"
            )
        return INCIDENT_TYPE_CONFIGS.get(incident_type_key)

    def _get_random_timestamp_for_day(self, value: datetime) -> datetime:
        start_of_day = value.replace(
            hour=0, minute=0, second=0, microsecond=000000
        )
        minutes_offset = random.randrange(0, 60 * 24)
        return start_of_day + timedelta(minutes=minutes_offset)

    def _get_incident_type(self, incident_type_key: str) -> IncidentType:
        return self._incident_type_map.get(incident_type_key.upper())

    def _get_random_priority(self) -> Incident.Priority:
        return random.choice(
            # 10% chance of high
            [Incident.Priority.HIGH]
            # 30% chance of medium
            + [Incident.Priority.MEDIUM] * 3
            # 60% chance of low
            + [Incident.Priority.LOW] * 6
        )

    def _delete_existing_demo_incidents(
        self, config: DemoIncidentTypeConfig
    ) -> None:
        existing_demo_incidents = Incident.objects.filter(
            organization__key=DEMO_ORG_KEY,
            zone__key=DEMO_SITE_KEY,
            incident_type__key__iexact=config.incident_type_key,
        )

        if existing_demo_incidents:
            logger.warning(
                f"Deleting {len(existing_demo_incidents)} existing demo incidents"
            )
            existing_demo_incidents.delete()
        else:
            logger.info("No demo incidents found")

    def _ensure_incident_type_exists(
        self, config: DemoIncidentTypeConfig
    ) -> None:
        try:
            incident_type = IncidentType.objects.get(
                key__iexact=config.incident_type_key
            )
        except IncidentType.DoesNotExist as error:
            raise RuntimeError(
                f"No incident type found for key: {config.incident_type_key}"
            ) from error

        OrganizationIncidentType.objects.update_or_create(
            organization=self._organization,
            incident_type=incident_type,
            defaults={
                "enabled": True,
            },
        )

        SiteIncidentType.objects.update_or_create(
            site=self._site,
            incident_type=incident_type,
            defaults={
                "enabled": True,
            },
        )

    def _ensure_source_incidents_exist(
        self, config: DemoIncidentTypeConfig
    ) -> None:
        missing_uuids = config.source_incident_uuids.copy()
        source_incidents = Incident.objects_raw.filter(
            uuid__in=config.source_incident_uuids
        )

        for source_incident in source_incidents:
            missing_uuids.remove(str(source_incident.uuid))

        if missing_uuids:
            raise RuntimeError(
                f"Source incidents do not exist: {missing_uuids}"
            )

    def _create_or_update_demo_camera(self, source_camera: Camera) -> Camera:
        demo_camera_uuid = f"{DEMO_CAMERA_UUID_PREFIX}/{source_camera.uuid}"
        demo_camera, _ = Camera.objects.update_or_create(
            uuid=demo_camera_uuid,
            defaults={
                "organization": self._organization,
                "zone": self._site,
                "name": source_camera.name,
                # Reset the thumbnail path so that it gets re-generated
                "thumbnail_s3_path": None,
            },
        )
        return demo_camera

    def _create_or_update_camera_incident_type(
        self, camera: Camera, incident_type: IncidentType
    ) -> None:
        CameraIncidentType.objects.update_or_create(
            camera=camera,
            incident_type=incident_type,
            defaults={
                "enabled": True,
            },
        )

    def _copy_s3_object(
        self,
        incident_uuid: str,
        object_suffix: str,
        source_s3_path: str,
    ) -> str:
        bucket_name, _ = separate_bucket_from_relative_path(source_s3_path)
        _, extension = os.path.splitext(source_s3_path)
        dest_object_name = f"{incident_uuid}_{object_suffix}{extension}"
        directory = f"{DEMO_ORG_KEY.lower()}/{DEMO_SITE_KEY.lower()}/incidents"
        dest_s3_path = f"s3://{bucket_name}/{directory}/{dest_object_name}"
        copy_object(source_s3_path, dest_s3_path)
        return dest_s3_path

    def _create_demo_incident(
        self,
        source_incident_uuid: str,
        timestamp: datetime,
        incident_type_key: str,
    ) -> None:
        source_incident = Incident.objects_raw.get(uuid=source_incident_uuid)
        source_data = source_incident.data
        new_uuid = str(uuid4())

        # Make copies of S3 objects
        video_s3_path = self._copy_s3_object(
            new_uuid,
            "video",
            source_data.get("video_s3_path"),
        )
        thumbnail_s3_path = self._copy_s3_object(
            new_uuid,
            "thumbnail",
            source_data.get("video_thumbnail_s3_path"),
        )
        annotations_s3_path = self._copy_s3_object(
            new_uuid,
            "annotations",
            source_data.get("annotations_s3_path"),
        )

        incident_type = self._get_incident_type(incident_type_key)

        if source_incident.camera:
            camera = self._create_or_update_demo_camera(source_incident.camera)
            self._create_or_update_camera_incident_type(camera, incident_type)
        else:
            camera = None
            logger.warning(
                f"No camera defined for source incident: {source_incident_uuid}"
            )

        Incident.objects.create(
            uuid=new_uuid,
            timestamp=timestamp,
            title=incident_type.name,
            priority=self._get_random_priority(),
            review_level=ReviewLevel.GOLD,
            review_status=ReviewStatus.VALID,
            visible_to_customers=True,
            experimental=False,
            incident_type=incident_type,
            organization=self._organization,
            zone=self._site,
            camera=camera,
            data={
                "video_s3_path": video_s3_path,
                "video_thumbnail_s3_path": thumbnail_s3_path,
                "annotations_s3_path": annotations_s3_path,
                "start_frame_relative_ms": source_data.get(
                    "start_frame_relative_ms"
                ),
                "end_frame_relative_ms": source_data.get(
                    "end_frame_relative_ms"
                ),
                "actor_ids": source_data.get("actor_ids"),
            },
        )

        logger.info(
            f"Created new demo incident: https://app.voxelai.com/incidents/{new_uuid}"
        )

    def _create_demo_incidents(self, config: DemoIncidentTypeConfig) -> None:
        now = datetime.now(self._organization.tzinfo)

        for day_offset, incident_count in config.relative_day_config.items():
            for _ in range(incident_count):
                current_day = now + timedelta(days=day_offset)
                timestamp = self._get_random_timestamp_for_day(current_day)
                source_incident_uuid = config.source_uuid_queue.get_value()
                self._create_demo_incident(
                    source_incident_uuid,
                    timestamp,
                    config.incident_type_key,
                )

    def run(self, incident_type_key: str):
        """Run the demo data generator.

        Args:
            incident_type_key (str): incident type key to generate data for

        Raises:
            RuntimeError: if the config is invalid
        """
        config = self._get_config(incident_type_key)
        if not config.is_valid():
            raise RuntimeError(
                f"Invalid config for incident type: {incident_type_key}"
            )

        self._ensure_incident_type_exists(config)
        self._ensure_source_incidents_exist(config)
        self._delete_existing_demo_incidents(config)
        self._create_demo_incidents(config)


def parse_args() -> Arguments:
    """Parse CLI arguments.

    Returns:
        Arguments: argparse arguments
    """
    parser = argparse.ArgumentParser(description="Demo Data Generator")
    parser.add_argument("-t", "--incident_type_key", type=str, required=True)
    return Arguments(**vars(parser.parse_args()))


if __name__ == "__main__":
    args = parse_args()
    DemoDataGenerator().run(args.incident_type_key)
