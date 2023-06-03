import os
import random
import time
import uuid
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import factory
from django.contrib.admin.models import LogEntry as DjangoAdminLogEntry
from django.contrib.auth.models import Group, User
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.portal.accounts.models.user_role import UserRole
from core.portal.accounts.roles import EXTERNAL_MANAGER, sync_static_roles
from core.portal.activity.enums import SessionScope
from core.portal.activity.models.user_session import UserSession
from core.portal.analytics.enums import AggregateGroup
from core.portal.api.models.comment import Comment
from core.portal.api.models.incident import Incident, UserIncident
from core.portal.api.models.incident_feedback import IncidentFeedback
from core.portal.api.models.incident_type import (
    CameraIncidentType,
    IncidentType,
)
from core.portal.api.models.invitation import Invitation
from core.portal.api.models.list import List as ListModel
from core.portal.api.models.notification_log import NotificationLog
from core.portal.api.models.organization import Organization
from core.portal.api.models.profile import Profile
from core.portal.api.models.share_link import ShareLink
from core.portal.compliance.enums import ComplianceTypeKey
from core.portal.compliance.models.compliance_type import ComplianceType
from core.portal.compliance.models.door_event_aggregate import (
    DoorEventAggregate,
)
from core.portal.compliance.models.door_open_aggregate import DoorOpenAggregate
from core.portal.compliance.models.production_line import ProductionLine
from core.portal.compliance.models.production_line_aggregate import (
    ProductionLineAggregate,
)
from core.portal.compliance.models.zone_compliance_type import (
    ZoneComplianceType,
)
from core.portal.devices.edge_lifecycle import STATIC_EDGE_LIFECYCLE_CONFIG_MAP
from core.portal.devices.models.camera import Camera, CameraConfigNew
from core.portal.devices.models.edge import Edge
from core.portal.incidents.enums import IncidentCategory, ReviewStatus
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.lib.enums import TimeBucketWidth
from core.portal.notifications.enums import NotificationCategory
from core.portal.perceived_data.enums import (
    PerceivedActorStateDurationCategory,
)
from core.portal.perceived_data.models.perceived_actor_state_duration_aggregate import (
    PerceivedActorStateDurationAggregate,
)
from core.portal.perceived_data.models.perceived_event_rate_hourly import (
    PerceivedEventRateDefinition,
    PerceivedEventRateHourly,
)
from core.portal.state.models.event import Event
from core.portal.state.models.state import State
from core.portal.testing.factories import (
    CameraConfigNewFactory,
    CameraFactory,
    CommentFactory,
    ComplianceTypeFactory,
    DoorEventAggregateFactory,
    DoorOpenAggregateFactory,
    EventFactory,
    GroupFactory,
    IncidentFactory,
    IncidentTypeFactory,
    NotificationLogFactory,
    OrganizationFactory,
    PerceivedActorStateDurationAggregateFactory,
    ProductionLineAggregateFactory,
    ProductionLineFactory,
    RoleFactory,
    StateFactory,
    UserFactory,
    UserRoleFactory,
    UserSessionFactory,
    ZoneComplianceTypeFactory,
    ZoneFactory,
)
from core.portal.testing.test_users import (
    REVIEW_MANAGER,
    REVIEWER,
    SITE_ADMIN,
    SITE_MANAGER,
    SUPERUSER,
    sync_test_user,
)
from core.portal.testing.utils import random_datetime
from core.portal.utils.enum_utils import (
    fetch_enum_model_from_name,
    sync_static_enum,
)
from core.portal.zones.enums import ZoneType
from core.portal.zones.models.zone import Zone
from core.structs.actor import ActorCategory
from core.structs.ergonomics import PostureType

# trunk-ignore-all(bandit/B311): nothing here requires cryptographic strength
# trunk-ignore-all(pylint/R0915,pylint/R0912): the big nasty seed function is ok for now

MODE_REFRESH = "refresh"  # Clear all data, create new data
MODE_CLEAR = "clear"  # Clear all data, don't create new data
TARGET_TOTAL_INCIDENT_COUNT = 100_000
SECONDS_IN_AN_HOUR = 3600


@dataclass
class OrganizationConfig:
    key: str
    name: str
    zones_per_org: int = 5
    cameras_per_zone: int = 5
    users_per_zone: int = 5
    incidents_per_camera: int = 25
    events_per_camera: int = 1000
    ergonomics_state_per_camera: int = 1000
    production_lines_per_camera: int = 0
    data_start_date: datetime = timezone.now() - timedelta(days=32)
    data_end_date: datetime = timezone.now()

    @property
    def organization(self) -> Organization:
        """Organization instance.

        Returns:
            Organization: organization
        """
        return Organization.objects.get(key=self.key)

    def org_site_camera_tuples(
        self,
    ) -> List[Tuple[Organization, Zone, Camera]]:
        """Get list of all org/zone/camera tuples.

        Returns:
            Tuple[Organization, Zone, Camera]: list of tuples
        """
        tuples = []
        for site in self.organization.sites.all():
            for camera in site.cameras.all():
                tuples.append((self.organization, site, camera))
        return tuples

    def random_start_end_timestamps(
        self,
        duration_min_s: int = 1,
        duration_max_s: int = 30,
    ) -> Tuple[datetime, datetime]:
        """Generate random start/end timestamp pair.

        Args:
            duration_min_s (int, optional): minimum duration in seconds. Defaults to 1.
            duration_max_s (int, optional): maximum duration in seconds. Defaults to 30.

        Returns:
            Tuple[datetime, datetime]: start/end timestamp pair
        """
        duration_s = random.randint(duration_min_s, duration_max_s)
        start = random_datetime(self.data_start_date, self.data_end_date)
        end = start + timedelta(seconds=duration_s)
        return (start, end)


ORGANIZATION_CONFIGS = [
    OrganizationConfig(
        "ACME",
        "ACME Logistics",
        production_lines_per_camera=2,
    ),
    OrganizationConfig("TPM_GLOBAL", "TPM Shipping & Storage"),
]


class Command(BaseCommand):
    help = "seed database for testing and development."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--mode", type=str, help="Mode")

    def handle(self, *_: None, **options: str) -> None:
        print("Starting seed process...")
        run_seed(options["mode"])
        print("Done")


def clear_data() -> None:
    """Clear all data."""

    querysets = [
        DjangoAdminLogEntry.objects.all(),
        NotificationLog.objects.all(),
        ShareLink.objects.all(),
        Invitation.zones.through.objects.all(),
        Invitation.objects.all(),
        UserIncident.objects.all(),
        ListModel.incidents.through.objects.all(),
        ListModel.objects.all(),
        Profile.objects.all(),
        Comment.objects.all(),
        IncidentFeedback.objects.all(),
        Incident.objects_raw.all(),
        Zone.users.through.objects.all(),
        Organization.users.through.objects.all(),
        Group.permissions.through.objects.all(),
        Group.user_set.through.objects.all(),
        Group.objects.all(),
        User.user_permissions.through.objects.all(),
        User.auth_token.get_queryset(),
        UserRole.objects.all(),
        UserSession.objects.all(),
        User.objects.all(),
        DoorEventAggregate.objects.all(),
        ZoneComplianceType.objects.all(),
        Edge.objects.all(),
        DoorOpenAggregate.objects.all(),
        ProductionLineAggregate.objects.all(),
        ProductionLine.objects.all(),
        PerceivedActorStateDurationAggregate.objects.all(),
        PerceivedEventRateHourly.objects.all(),
        PerceivedEventRateDefinition.objects.all(),
        CameraIncidentType.objects.all(),
        CameraConfigNew.objects.all(),
        Camera.objects.all(),
        Camera.incident_types.through.objects.all(),
        Zone.objects.all(),
        Organization.incident_types.through.objects.all(),
        Organization.objects.all(),
        IncidentType.objects.all(),
        Group.objects.all(),
        ComplianceType.objects.all(),
        ZoneComplianceType.objects.all(),
        Event.objects.all(),
        State.objects.all(),
    ]

    for queryset in querysets:
        # trunk-ignore(pylint/W0212): Django doesn't provide a better bulk delete method :(
        queryset._raw_delete(queryset.db)


def seed_ergonomic_state_data(config: OrganizationConfig) -> None:
    """Create ergonomic state data for the provided organization config.

    Args:
        config (OrganizationConfig): organization config
    """
    objects = []
    for org, site, camera in config.org_site_camera_tuples():
        for _ in range(config.ergonomics_state_per_camera):
            start, end = config.random_start_end_timestamps()
            objects.append(
                StateFactory.build(
                    timestamp=start,
                    end_timestamp=end,
                    organization=org.key,
                    location=site.key,
                    camera_uuid=camera.uuid,
                    person_lift_type=random.choice(
                        [PostureType.GOOD.value, PostureType.BAD.value]
                    ),
                )
            )
    State.objects.bulk_create(objects)


def seed_event_data(config: OrganizationConfig) -> None:
    """Create event data for the provided organization config.

    Args:
        config (OrganizationConfig): organization config
    """
    objects = []
    for org, site, camera in config.org_site_camera_tuples():
        for _ in range(config.events_per_camera):
            start, end = config.random_start_end_timestamps()
            objects.append(
                EventFactory.build(
                    timestamp=start,
                    end_timestamp=end,
                    organization=org.key,
                    location=site.key,
                    camera_uuid=camera.uuid,
                )
            )
    Event.objects.bulk_create(objects)


def seed_door_event_aggregate_data(config: OrganizationConfig) -> None:
    """Create door event aggregate data for the provided organization config.

    Args:
        config (OrganizationConfig): organization config
    """
    objects = []
    for org, site, camera in config.org_site_camera_tuples():
        current_date = config.data_start_date.replace(
            minute=0, second=0, microsecond=0
        )
        while current_date < config.data_end_date:
            opened_count = random.randint(0, 100)
            objects.append(
                DoorEventAggregateFactory.build(
                    group_key=current_date,
                    group_by=AggregateGroup.HOUR,
                    organization=org,
                    zone=site,
                    camera=camera,
                    opened_count=opened_count,
                    closed_within_30_seconds_count=int(opened_count * 0.3),
                    closed_within_1_minute_count=int(opened_count * 0.2),
                    closed_within_5_minutes_count=int(opened_count * 0.1),
                    closed_within_10_minutes_count=int(opened_count * 0.05),
                )
            )
            current_date = current_date + timedelta(hours=1)
    DoorEventAggregate.objects.bulk_create(objects)


def seed_door_open_aggregate_data(config: OrganizationConfig) -> None:
    """Create door open aggregate data for provided organization config

    Args:
        config (OrganizationConfig): organization config
    """
    objects = []

    for org, site, camera in config.org_site_camera_tuples():
        current_datetime = config.data_start_date.replace(
            minute=0, second=0, microsecond=0
        )
        while current_datetime < config.data_end_date:
            # Randomly decided to set the open and partial open time to 1/10 so that the close
            # time is substantially more significant than the open and partial open time.
            open_time_duration_s = random.randint(0, SECONDS_IN_AN_HOUR / 10)
            partially_open_time_duration_s = random.randint(
                0, SECONDS_IN_AN_HOUR / 10
            )
            close_time_duration_s = random.randint(
                open_time_duration_s + partially_open_time_duration_s,
                SECONDS_IN_AN_HOUR,
            )

            objects.append(
                DoorOpenAggregateFactory.build(
                    group_key=current_datetime,
                    group_by=AggregateGroup.HOUR,
                    organization=org,
                    zone=site,
                    camera=camera,
                    open_time_duration_s=open_time_duration_s,
                    close_time_duration_s=close_time_duration_s,
                    partially_open_time_duration_s=partially_open_time_duration_s,
                )
            )
            current_datetime += timedelta(hours=1)
    DoorOpenAggregate.objects.bulk_create(objects)


def seed_production_line_state_data(config: OrganizationConfig) -> None:
    """Create production line state data for the provided organization config.

    For each production line, we want to simulate a blend of uptime, downtime,
    and "unknown" time. In production, it's possible that cameras are unavailable
    or some other outage results in a gap in state messages, so occassional gaps
    in state are expected.

    Args:
        config (OrganizationConfig): organization config
    """
    objects = []
    for org, site, camera in config.org_site_camera_tuples():
        for production_line in camera.production_lines.all():
            current_datetime = config.data_start_date.replace(
                minute=0, second=0, microsecond=0
            )
            uptime = True
            while current_datetime < config.data_end_date:
                # Random duration between 1 second and 2 hours
                duration_s = random.randint(1, 7200)
                end_datetime = current_datetime + timedelta(seconds=duration_s)

                # 95% of the time we have valid uptime/downtime state
                # and the other 5% of the time we skip adding the state but
                # still increment current_datetime to simulate missing data
                if random.randint(0, 100) > 95:
                    objects.append(
                        StateFactory.build(
                            timestamp=current_datetime,
                            end_timestamp=end_datetime,
                            organization=org.key,
                            location=site.key,
                            camera_uuid=camera.uuid,
                            actor_category=ActorCategory.MOTION_DETECTION_ZONE.value,
                            actor_id=str(production_line.uuid),
                            motion_zone_is_in_motion=uptime,
                        )
                    )
                    # Invert the uptime status
                    uptime = not uptime
                # Always increment current_datetime
                current_datetime = end_datetime
    State.objects.bulk_create(objects)


def seed_production_line_down_incidents(config: OrganizationConfig) -> None:
    """Create production line down incident data for the provided organization config.

    Args:
        config (OrganizationConfig): organization config
    """
    incident_type = IncidentTypeFactory(key="PRODUCTION_LINE_DOWN")
    objects = []
    for org, site, camera in config.org_site_camera_tuples():
        for production_line in camera.production_lines.all():
            current_datetime = config.data_start_date.replace(
                minute=0, second=0, microsecond=0
            )
            while current_datetime < config.data_end_date:
                # Seed between 0-10 incidents per hour
                for _ in range(random.randint(0, 10)):
                    timestamp = current_datetime + timedelta(
                        minutes=random.randint(0, 59)
                    )
                    incident = IncidentFactory.build(
                        timestamp=timestamp,
                        organization=org,
                        zone=site,
                        camera=camera,
                        incident_type=incident_type,
                        visible_to_customers=True,
                        review_level=ReviewLevel.GOLD,
                        review_status=ReviewStatus.VALID,
                    )
                    incident.data["track_uuid"] = production_line.uuid
                    objects.append(incident)
                current_datetime += timedelta(hours=1)
    Incident.objects.bulk_create(objects)


def seed_production_line_event_aggregate_data(
    config: OrganizationConfig,
) -> None:
    """Create production line aggregate data for the provided organization config

    Args:
        config (OrganizationConfig): organization config
    """

    objects = []

    for org, site, camera in config.org_site_camera_tuples():
        for production_line in camera.production_lines.all():
            current_date = config.data_start_date.replace(
                minute=0, second=0, microsecond=0
            )
            while current_date < config.data_end_date:
                uptime_duration_s = random.randint(0, SECONDS_IN_AN_HOUR)
                downtime_duration_s = random.randint(
                    0, SECONDS_IN_AN_HOUR - uptime_duration_s
                )
                objects.append(
                    ProductionLineAggregateFactory.build(
                        group_key=current_date,
                        group_by=AggregateGroup.HOUR,
                        organization=org,
                        zone=site,
                        camera=camera,
                        production_line=production_line,
                        uptime_duration_s=uptime_duration_s,
                        downtime_duration_s=downtime_duration_s,
                    )
                )
                current_date = current_date + timedelta(hours=1)
    ProductionLineAggregate.objects.bulk_create(objects)


def seed_perceived_actor_state_duration_aggregate_data(
    config: OrganizationConfig,
):
    """Create perceived actor state duration aggregate data for provided organization config.

    Args:
        config (OrganizationConfig): organization config
    """
    objects = []

    for _, _, camera in config.org_site_camera_tuples():
        current_timestamp = config.data_start_date.replace(
            minute=0, second=0, microsecond=0
        )
        while current_timestamp < config.data_end_date:
            for category_value in PerceivedActorStateDurationCategory.values:
                objects.append(
                    PerceivedActorStateDurationAggregateFactory.build(
                        time_bucket_start_timestamp=current_timestamp,
                        time_bucket_width=TimeBucketWidth.HOUR,
                        category=category_value,
                        camera_uuid=camera.uuid,
                    )
                )
            current_timestamp += timedelta(hours=1)
    PerceivedActorStateDurationAggregate.objects.bulk_create(objects)


def seed_activity_session_data(config: OrganizationConfig) -> None:
    """Create activity session data for the provided organization config.

    Args:
        config (OrganizationConfig): organization config
    """

    objects = []

    for site in config.organization.sites.all():
        for user in site.users.all():
            session_key = UserSession.build_key(
                user.id,
                scope=SessionScope.SITE,
                scope_id=site.id,
                start_timestamp=timezone.now(),
            )
            objects.append(
                UserSessionFactory.build(
                    start_timestamp=timezone.now(),
                    end_timestamp=timezone.now() + timedelta(minutes=30),
                    site=site,
                    user=user,
                    key=session_key,
                )
            )

    UserSession.objects.bulk_create(objects)


def run_seed(mode: str) -> None:
    """Seed database based on mode."""
    if os.getenv("ENVIRONMENT") not in ("development", "test"):
        raise RuntimeError(
            "Seed script is only allowed in dev/test environments"
        )

    clear_data()
    if mode == MODE_CLEAR:
        return

    start_time = time.time()

    # **************************************************************************
    # Update all static enums:
    # 1. Roles
    # 2. Edge Lifecycle
    # **************************************************************************
    sync_static_roles()
    sync_static_enum(
        fetch_enum_model_from_name("EdgeLifecycle"),
        STATIC_EDGE_LIFECYCLE_CONFIG_MAP,
    )

    # **************************************************************************
    # Incident types
    # **************************************************************************
    print("Seeding incident types...")
    all_incident_types = [
        IncidentTypeFactory(
            key="OPEN_DOOR_DURATION",
            value="OPEN_DOOR_DURATION",
            name="Open Door Duration",
            background_color="#ecd361",
            category=IncidentCategory.ENVIRONMENT.value,
        ),
        IncidentTypeFactory(
            key="NO_STOP_AT_INTERSECTION",
            value="NO_STOP_AT_INTERSECTION",
            name="No Stop At Intersection",
            background_color="#8a291d",
            category=IncidentCategory.VEHICLE.value,
        ),
        IncidentTypeFactory(
            key="PIGGYBACK",
            value="PIGGYBACK",
            name="Piggyback",
            background_color="#734c83",
            category=IncidentCategory.VEHICLE.value,
        ),
        IncidentTypeFactory(
            key="PARKING_DURATION",
            value="PARKING_DURATION",
            name="Parking Duration",
            background_color="#2980b9",
            category=IncidentCategory.VEHICLE.value,
        ),
        IncidentTypeFactory(
            key="DOOR_VIOLATION",
            value="DOOR_VIOLATION",
            name="Door Violation",
            background_color="#16a085",
            category=IncidentCategory.ENVIRONMENT.value,
        ),
        IncidentTypeFactory(
            key="MISSING_PPE",
            value="MISSING_PPE",
            name="Missing PPE",
            background_color="#dd7700",
            category=IncidentCategory.PEOPLE.value,
        ),
        IncidentTypeFactory(
            key="BAD_POSTURE",
            value="BAD_POSTURE",
            name="Bad Posture",
            background_color="#ff9900",
            category=IncidentCategory.PEOPLE.value,
        ),
        IncidentTypeFactory(
            key="PRODUCTION_LINE_DOWN",
            value="PRODUCTION_LINE_DOWN",
            name="Production Line Down",
            background_color="#16a085",
            category=IncidentCategory.ENVIRONMENT.value,
        ),
    ]

    # **************************************************************************
    # Compliance types
    # **************************************************************************
    for (key, name) in ComplianceTypeKey.choices:
        ComplianceTypeFactory(key=key, name=name)
    all_compliance_types = ComplianceType.objects.all()

    # **************************************************************************
    # Organizations
    # **************************************************************************
    for config in ORGANIZATION_CONFIGS:
        print(f"Seeding organization: {config.name}...")

        org = OrganizationFactory(
            name=config.name,
            key=config.key,
            incident_types=all_incident_types,
        )

        for _ in range(config.zones_per_org):
            zone = ZoneFactory(organization=org, zone_type=ZoneType.SITE)

            for i in range(config.users_per_zone):
                user = UserFactory(
                    email=f"staff_{zone.key}_{i+1}@example.com",
                    groups=[
                        GroupFactory(name="staff"),
                    ],
                )
                UserRoleFactory(
                    user=user,
                    role=RoleFactory(key=EXTERNAL_MANAGER, name="Manager"),
                )
                user.profile.organization = org
                user.profile.site = zone
                user.profile.save()
                zone.users.add(user)

                # TODO: don't add user to org list, once zone filtering is in place
                org.users.add(user)

            for _ in range(config.cameras_per_zone):
                camera = CameraFactory(
                    organization=org,
                    zone=zone,
                    incident_types=all_incident_types,
                )
                CameraConfigNewFactory(camera=camera, version=1)

                for _ in range(config.incidents_per_camera):
                    # Standard incidents
                    incident = IncidentFactory(
                        organization=org,
                        valid_feedback_count=1,
                        visible_to_customers=True,
                        review_level=ReviewLevel.GOLD,
                        review_status=ReviewStatus.VALID,
                        timestamp=random_datetime(
                            config.data_start_date, config.data_end_date
                        ),
                        camera=camera,
                        zone=zone,
                    )

                    for _ in range(random.randint(0, 2)):
                        CommentFactory(
                            incident=incident,
                            owner=factory.Iterator(zone.users.all()),
                        )

                    # Seed some alerts
                    if random.randint(0, 10) == 0:
                        NotificationLogFactory(
                            user=random.choice(zone.users.all()),
                            incident=incident,
                            sent_at=timezone.now(),
                            category=NotificationCategory.INCIDENT_ALERT,
                        )
                        incident.alerted = True
                        incident.save()

                    # Experimental incidents
                    IncidentFactory(
                        experimental=True,
                        visible_to_customers=False,
                        timestamp=random_datetime(
                            config.data_start_date, config.data_end_date
                        ),
                        incident_version="experimental-seed",
                        camera=camera,
                        zone=zone,
                    )

                # Production lines
                for i in range(config.production_lines_per_camera):
                    ProductionLineFactory(
                        name=f"{camera.name} - Production Line #{i}",
                        organization=org,
                        zone=zone,
                        camera=camera,
                    )
            # Compliance types
            for compliance_type in all_compliance_types:
                ZoneComplianceTypeFactory(
                    zone=zone, compliance_type=compliance_type, enabled=True
                )
        seed_event_data(config)
        seed_ergonomic_state_data(config)
        seed_door_event_aggregate_data(config)
        seed_production_line_state_data(config)
        seed_production_line_event_aggregate_data(config)
        seed_production_line_down_incidents(config)
        seed_door_open_aggregate_data(config)
        seed_perceived_actor_state_duration_aggregate_data(config)
        seed_activity_session_data(config)

    # **************************************************************************
    # Duplicate incidents up to the desired total count
    # **************************************************************************
    print("Duplicating incidents to simulate more data volume...")
    batch_size = 500
    offset = 0
    current_incident_count = Incident.objects.count()
    while current_incident_count < TARGET_TOTAL_INCIDENT_COUNT:
        incidents = Incident.objects.order_by("timestamp")[:batch_size]
        for incident in incidents:
            incident.pk = None
            incident.uuid = uuid.uuid4()

        Incident.objects.bulk_create(list(incidents))
        current_incident_count = Incident.objects.count()
        percent_complete = int(
            (current_incident_count / TARGET_TOTAL_INCIDENT_COUNT) * 100
        )
        print(f"Progress: {percent_complete}%", end="\r", flush=True)

        # Reset offset if offset is greater than total incident count
        offset = 0 if offset >= current_incident_count else offset + batch_size

    # **************************************************************************
    # Users
    # **************************************************************************
    print("Seeding static users...")
    initial_org = Organization.objects.get(key=ORGANIZATION_CONFIGS[0].key)
    initial_site = initial_org.sites.first()

    sync_test_user(SUPERUSER, [initial_org], [initial_site])
    sync_test_user(SITE_ADMIN, [initial_org], [initial_site])
    sync_test_user(SITE_MANAGER, [initial_org], [initial_site])
    sync_test_user(REVIEWER)
    sync_test_user(REVIEW_MANAGER)

    # Auth0 M2M
    # https://manage.auth0.com/dashboard/us/voxeldev/applications/7aHXbzZYsQcGYcFaudPWO1qc5Ab8VZDs/settings
    perception_m2m = UserFactory(
        username="7aHXbzZYsQcGYcFaudPWO1qc5Ab8VZDs@clients",
        email="perception_m2m@example.com",
        is_staff=True,
        is_superuser=True,
    )
    perception_m2m.profile.organziation = initial_org
    perception_m2m.profile.site = initial_site
    perception_m2m.profile.save()

    initial_org.users.add(perception_m2m)
    initial_site.users.add(perception_m2m)

    # **************************************************************************
    # Finished
    # **************************************************************************
    end_time = time.time()
    duration = str(timedelta(seconds=(end_time - start_time)))
    print(f"Finished (seed time: {duration})")
