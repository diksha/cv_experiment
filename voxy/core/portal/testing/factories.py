import copy
import random
import string
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Type, Union
from uuid import uuid4

import factory
import factory.fuzzy
import pytz
from django.conf import settings
from django.contrib.auth.models import Group, Permission, User
from django.db.models.signals import post_save
from django.utils import timezone
from factory.django import DjangoModelFactory

from core.portal.accounts.models.role import Role
from core.portal.accounts.models.role_permission import RolePermission
from core.portal.accounts.models.user_role import UserRole
from core.portal.activity.models.user_session import UserSession
from core.portal.analytics.enums import AggregateGroup
from core.portal.api.models.comment import Comment
from core.portal.api.models.incident import Incident
from core.portal.api.models.incident_feedback import IncidentFeedback
from core.portal.api.models.incident_type import (
    CameraIncidentType,
    IncidentType,
    OrganizationIncidentType,
)
from core.portal.api.models.invitation import Invitation
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
from core.portal.compliance.models.production_line_aggregate import (
    ProductionLine,
    ProductionLineAggregate,
)
from core.portal.compliance.models.zone_compliance_type import (
    ZoneComplianceType,
)
from core.portal.devices.models.camera import Camera, CameraConfigNew
from core.portal.incidents.enums import ReviewStatus
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.lib.enums import TimeBucketWidth
from core.portal.notifications.enums import NotificationCategory
from core.portal.perceived_data.models.perceived_actor_state_duration_aggregate import (
    PerceivedActorStateDurationAggregate,
)
from core.portal.state.models.event import Event
from core.portal.state.models.state import State
from core.portal.testing.utils import random_datetime, strip_non_alphanumeric
from core.portal.zones.enums import ZoneType
from core.portal.zones.models import Zone, ZoneUser
from core.structs.actor import ActorCategory
from core.structs.ergonomics import ActivityType, PostureType
from core.structs.event import EventType

# trunk-ignore-all(pylint/E1101): factory_boy is doing a lot of dynamic stuff
# trunk-ignore-all(bandit/B311): nothing here requires cryptographic strength
# trunk-ignore-all(pylint/C0301): long lines allowed for data configurations

if settings.STAGING:
    S3_BUCKET = "s3://voxel-portal-staging"
elif settings.DEVELOPMENT or settings.TEST:
    S3_BUCKET = "s3://voxel-portal-dev"
else:
    raise RuntimeError(
        "factories should not be imported outside of dev/test/staging"
    )


def enum_values(enum_class: Type[Enum]) -> List[Union[str, int]]:
    """Convert enum members to list of values.

    Args:
        enum_class (Type[Enum]): enum class

    Returns:
        List[Union[str, int]]: list of enum values
    """
    return [e.value for e in enum_class]


@factory.django.mute_signals(post_save)
class ProfileFactory(DjangoModelFactory):
    class Meta:
        model = Profile
        django_get_or_create = ["owner"]

    data = {}
    organization = factory.SubFactory(
        "core.portal.testing.factories.OrganizationFactory"
    )
    owner = factory.SubFactory("testing.factories.UserFactory", profile=None)


@factory.django.mute_signals(post_save)
class UserFactory(DjangoModelFactory):
    class Meta:
        model = User
        django_get_or_create = ["email"]

    email = factory.Faker("ascii_company_email")
    first_name = factory.Faker("first_name")
    last_name = factory.Faker("last_name")
    username = factory.SelfAttribute("email")
    is_active = True
    is_superuser = False
    is_staff = False
    profile = factory.RelatedFactory(
        ProfileFactory, factory_related_name="owner"
    )

    @factory.post_generation
    def permissions(
        self, create: bool, extracted: List[str], **_: None
    ) -> None:
        if not create:
            return
        if extracted:
            self.permissions = extracted
        else:
            self.permissions = []

    @factory.post_generation
    def groups(self, create: bool, extracted: List[str], **_: None) -> None:
        if not create:
            return
        if extracted:
            for group in extracted:
                self.groups.add(group)


class PermissionFactory(DjangoModelFactory):
    class Meta:
        model = Permission
        django_get_or_create = ["codename"]


class GroupFactory(DjangoModelFactory):
    class Meta:
        model = Group
        django_get_or_create = ["name"]

    @factory.post_generation
    def permissions(
        self, create: bool, extracted: List[str], **_: None
    ) -> None:
        if not create:
            return
        if extracted:
            for permission in extracted:
                self.permissions.add(permission)


class OrganizationFactory(DjangoModelFactory):
    class Meta:
        model = Organization
        django_get_or_create = ["name"]

    name = factory.Faker("company")

    @factory.post_generation
    def key(
        self: Organization, create: bool, extracted: str, **_: None
    ) -> None:
        if not create:
            return
        if extracted:
            self.key = extracted
        else:
            self.key = strip_non_alphanumeric(self.name.replace(" ", "_"))
        self.save()

    @factory.post_generation
    def users(
        self: Organization, create: bool, extracted: List[User], **_: None
    ) -> None:
        if not create:
            return
        if extracted:
            for user in extracted:
                self.users.add(user)

    @factory.post_generation
    def incident_types(
        self: Organization,
        create: bool,
        extracted: List[IncidentType],
        **_: None,
    ) -> None:
        if not create:
            return
        if extracted:
            for incident_type in extracted:
                self.incident_types.add(incident_type)

    @factory.post_generation
    def incidents(
        self: Organization, create: bool, extracted: List[Incident], **_: None
    ) -> None:
        if not create:
            return
        if extracted:
            for incident in extracted:
                self.incidents.add(incident)

    @factory.post_generation
    def zones(
        self: Organization, create: bool, extracted: List[Zone], **_: None
    ) -> None:
        if not create:
            return
        if extracted:
            for zone in extracted:
                self.zones.add(zone)

    @factory.post_generation
    def cameras(
        self: Organization, create: bool, extracted: List[Camera], **_: None
    ) -> None:
        """Helper method for defining cameras in OrganizationFactory
        Args:
            self (Organization): reference instance
            create (bool): whether it is created or not
            extracted (List[Camera]): the extracted camera
            **_: extra args
        """
        if not create:
            return
        if extracted:
            for camera in extracted:
                self.cameras.add(camera)


class ZoneFactory(DjangoModelFactory):
    class Meta:
        model = Zone

    organization = factory.SubFactory(OrganizationFactory)
    parent_zone = None
    name = factory.Faker("city")
    zone_type = factory.fuzzy.FuzzyChoice(ZoneType.values)
    timezone = factory.fuzzy.FuzzyChoice(
        [
            None,
            "US/Alaska",
            "US/Arizona",
            "US/Central",
            "US/Eastern",
            "US/Hawaii",
            "US/Mountain",
            "US/Pacific",
        ]
    )
    active = True

    @factory.post_generation
    def cameras(
        self: Zone, create: bool, extracted: List[Camera], **_: None
    ) -> None:
        if not create:
            return
        if extracted:
            for camera in extracted:
                self.cameras.add(camera)

    @factory.post_generation
    def key(self: Zone, create: bool, extracted: str, **_: None) -> None:
        if not create:
            return
        if extracted:
            self.key = extracted
        else:
            self.key = strip_non_alphanumeric(self.name.replace(" ", "_"))
        self.save()


class ZoneUserFactory(DjangoModelFactory):
    class Meta:
        model = ZoneUser

    user = factory.SubFactory(UserFactory)
    zone = factory.SubFactory(
        ZoneFactory,
    )
    is_assignable = True


class RoleFactory(DjangoModelFactory):
    class Meta:
        model = Role
        django_get_or_create = ["key"]

    key = factory.Faker("job")
    name = factory.Faker("job")
    visible_to_customers = True


class RolePermissionFactory(DjangoModelFactory):
    class Meta:
        model = RolePermission

    permission_key = factory.Faker("job")
    role = factory.SubFactory(RoleFactory)


class UserRoleFactory(DjangoModelFactory):
    class Meta:
        model = UserRole

    user = factory.SubFactory(UserFactory)
    role = factory.SubFactory(RoleFactory)


class InvitationFactory(
    DjangoModelFactory
):  # pylint: disable=too-few-public-methods
    """Invitation factory creates an instance of invitation for use in testing."""

    class Meta:  # pylint: disable=too-few-public-methods
        """Meta is metainfo on this class"""

        model = Invitation

    token = factory.LazyFunction(uuid4)
    expires_at = timezone.now() + timedelta(days=1)
    invitee = factory.SubFactory(UserFactory)
    invited_by = factory.SubFactory(UserFactory)
    role = factory.SubFactory(RoleFactory)
    redeemed = False
    organization = factory.SubFactory(OrganizationFactory)

    @factory.post_generation
    def zones(
        self: Invitation, create: bool, extracted: List[Zone], **_: None
    ) -> None:
        """Zones is a post gen fn to add zones to the new invitation instance.
        :param self: refers to class instance.
        :param create: boolean used internally
        :param extracted: extracted list of zones to add
        :param _: extra
        """
        if not create:
            return
        if extracted:
            for zone in extracted:
                self.zones.add(zone)


class IncidentTypeFactory(DjangoModelFactory):
    class Meta:
        model = IncidentType
        django_get_or_create = ["key"]

    key = "MISSING_PPE"
    value = factory.SelfAttribute("key")
    name = factory.SelfAttribute("key")
    background_color = "#ff0000"


class OrganizationIncidentTypeFactory(DjangoModelFactory):
    class Meta:
        """Factory meta class."""

        model = OrganizationIncidentType

    organization = factory.SubFactory(OrganizationFactory)
    incident_type = factory.SubFactory(IncidentTypeFactory)
    enabled = True


class ComplianceTypeFactory(DjangoModelFactory):
    """ComplianceType factory."""

    class Meta:
        """Meta class used for factory configuration."""

        model = ComplianceType
        django_get_or_create = ["key"]

    key = factory.fuzzy.FuzzyChoice(ComplianceTypeKey.choices)
    name = factory.LazyAttribute(lambda o: o.key[1])


class ZoneComplianceTypeFactory(DjangoModelFactory):
    """ZoneComplianceType factory."""

    class Meta:
        """Meta class used for factory configuration."""

        model = ZoneComplianceType
        django_get_or_create = ["zone", "compliance_type"]

    enabled = True
    name_override = None
    zone = factory.SubFactory(ZoneFactory)
    compliance_type = factory.SubFactory(ComplianceTypeFactory)


def _random_camera_name():
    prefixes = ["Camera", "Door", "Room", "Dock", "Hall"]
    suffixes = string.ascii_uppercase + string.digits
    return f"{random.choice(prefixes)} {random.choice(suffixes)}"


class CameraFactory(DjangoModelFactory):
    class Meta:
        """Factory meta class."""

        model = Camera

    uuid = factory.LazyFunction(uuid4)
    name = factory.LazyFunction(_random_camera_name)
    organization = factory.SubFactory(OrganizationFactory)
    zone = factory.SubFactory(ZoneFactory)

    @factory.post_generation
    def incident_types(
        self: Camera,
        create: bool,
        extracted: List[IncidentType],
        **_: None,
    ) -> None:
        """Helper method to add incident types to CameraFactory.
        Args:
            self (Camera): instance of Camera
            create (bool): whether created or not
            extracted (List[IncidentType]): extracted list of incident types
            **_: extra args
        """
        if not create:
            return
        if extracted:
            for incident_type in extracted:
                self.incident_types.add(incident_type)


class CameraConfigNewFactory(DjangoModelFactory):
    class Meta:
        """Factory meta class."""

        model = CameraConfigNew

    camera = factory.SubFactory(CameraFactory)


class CommentFactory(DjangoModelFactory):
    class Meta:
        model = Comment

    activity_type = Comment.ActivityType.COMMENT
    text = factory.Faker("text")

    @factory.post_generation
    def created_at(
        self: Comment, create: bool, extracted: datetime, **_: None
    ) -> None:
        if not create:
            return
        if not extracted:
            self.created_at = random_datetime(
                self.incident.timestamp, timezone.now()
            )
            self.save()


class IncidentFeedbackFactory(DjangoModelFactory):
    class Meta:
        model = IncidentFeedback


def data_factory():
    """Generate a random data from list.

    Returns:
        dict: dict of data
    """
    data_list = [
        {
            "uuid": "4e2aaad6-eca4-48a5-9a9c-7b750c1a8c1b",
            "actor_ids": "40773",
            "docker_image_tag": "buildkite_1637013530_022a04c1",
            "video_thumbnail_s3_path": f"{S3_BUCKET}/tpm_global/unknown/incidents/19311be4-5350-498c-8085-e96c5a6ef706_thumbnail.jpg",
            "annotations_s3_path": f"{S3_BUCKET}/tpm_global/unknown/incidents/19311be4-5350-498c-8085-e96c5a6ef706_annotations.json",
            "video_s3_path": f"{S3_BUCKET}/tpm_global/unknown/incidents/19311be4-5350-498c-8085-e96c5a6ef706_video.mp4",
            "camera_config_version": "1",
            "start_frame_relative_ms": f"{int(time.time()) + random.randint(1, 1000)}",
            "end_frame_relative_ms": f"{int(time.time()) + random.randint(1000, 1500)}",
        },
        {
            "uuid": "74a676e1-9f9c-40e0-b43f-8b90d2122600",
            "actor_ids": "9",
            "docker_image_tag": "buildkite_1637013530_022a04c1",
            "video_thumbnail_s3_path": f"{S3_BUCKET}/tpm_global/unknown/incidents/23ca7d2c-75a0-4b17-b906-cebf1b734854_thumbnail.jpg",
            "annotations_s3_path": f"{S3_BUCKET}/tpm_global/unknown/incidents/23ca7d2c-75a0-4b17-b906-cebf1b734854_annotations.json",
            "video_s3_path": f"{S3_BUCKET}/tpm_global/unknown/incidents/33276ff6-d136-49f9-a6a1-585a2b5a3220_video.mp4",
            "camera_config_version": "1",
            "start_frame_relative_ms": f"{int(time.time()) + random.randint(1, 1000)}",
            "end_frame_relative_ms": f"{int(time.time()) + random.randint(1000, 1500)}",
        },
        {
            "uuid": "8597bc49-c8bd-4932-9ea5-733a0e3c980d",
            "actor_ids": "5",
            "docker_image_tag": "buildkite_1637013530_022a04c1",
            "video_thumbnail_s3_path": f"{S3_BUCKET}/tpm_global/unknown/incidents/33276ff6-d136-49f9-a6a1-585a2b5a3220_thumbnail.jpg",
            "annotations_s3_path": f"{S3_BUCKET}/tpm_global/unknown/incidents/33276ff6-d136-49f9-a6a1-585a2b5a3220_annotations.json",
            "video_s3_path": f"{S3_BUCKET}/tpm_global/unknown/incidents/33276ff6-d136-49f9-a6a1-585a2b5a3220_video.mp4",
            "camera_config_version": "1",
            "start_frame_relative_ms": f"{int(time.time()) + random.randint(1, 1000)}",
            "end_frame_relative_ms": f"{int(time.time()) + random.randint(1000, 1500)}",
        },
    ]
    return copy.deepcopy(random.choice(data_list))


class IncidentFactory(DjangoModelFactory):
    class Meta:
        model = Incident

    uuid = factory.LazyFunction(uuid4)
    incident_type = factory.fuzzy.FuzzyChoice(IncidentType.objects.all())
    title = factory.SelfAttribute("incident_type.name")
    timestamp = timezone.now()
    organization = factory.SubFactory(OrganizationFactory)
    zone = factory.SubFactory(ZoneFactory)
    camera = factory.SubFactory(CameraFactory)
    visible_to_customers = False  # Matches prod behavior for new incidents
    review_level = ReviewLevel.RED
    review_status = ReviewStatus.NEEDS_REVIEW
    cooldown_source = None
    highlighted = False
    experimental = False  # Matches prod behavior for new incidents
    priority = factory.fuzzy.FuzzyChoice(Incident.Priority)
    data = factory.LazyFunction(data_factory)

    @factory.post_generation
    def incident_version(
        self, create: bool, extracted: str, **_: None
    ) -> None:
        if not create:
            return
        if extracted:
            self.data["incident_version"] = extracted


class NotificationLogFactory(DjangoModelFactory):
    class Meta:
        model = NotificationLog

    user = factory.SubFactory(UserFactory)
    category = factory.fuzzy.FuzzyChoice(NotificationCategory.choices)
    sent_at = datetime.now(tz=pytz.timezone("UTC"))
    incident = factory.SubFactory(IncidentFactory)
    data = factory.Dict({})


class DoorEventAggregateFactory(DjangoModelFactory):
    """Door event aggregate factory."""

    class Meta:
        """Factory meta class."""

        model = DoorEventAggregate

    group_key = factory.fuzzy.FuzzyDateTime(
        (timezone.now() - timedelta(days=30)).replace(
            minute=0, second=0, microsecond=0
        )
    )
    group_by = AggregateGroup.HOUR
    max_timestamp = factory.LazyAttribute(
        lambda o: o.group_key + timedelta(minutes=59)
    )

    organization = None
    zone = None
    camera = None

    opened_count = 0
    closed_within_30_seconds_count = 0
    closed_within_1_minute_count = 0
    closed_within_5_minutes_count = 0
    closed_within_10_minutes_count = 0


class ProductionLineFactory(DjangoModelFactory):
    """Production line factory."""

    class Meta:
        """Factory meta class."""

        model = ProductionLine

    uuid = factory.LazyFunction(uuid4)
    name = None
    organization = None
    zone = None
    camera = None


class UserSessionFactory(DjangoModelFactory):
    class Meta:
        model = UserSession

    start_timestamp = None
    end_timestamp = None
    site = None
    organization = None
    user = None


class ProductionLineAggregateFactory(DjangoModelFactory):
    """Production line aggregate factory."""

    class Meta:
        """Factory meta class."""

        model = ProductionLineAggregate

    group_key = factory.fuzzy.FuzzyDateTime(
        (timezone.now() - timedelta(days=30)).replace(
            minute=0, second=0, microsecond=0
        )
    )
    group_by = AggregateGroup.HOUR
    max_timestamp = factory.LazyAttribute(
        lambda o: o.group_key + timedelta(minutes=59)
    )

    organization = None
    zone = None
    camera = None
    production_line = None

    uptime_duration_s = 0
    downtime_duration_s = 0


class DoorOpenAggregateFactory(DjangoModelFactory):
    """Door open aggregate factory."""

    class Meta:
        """Factory meta class."""

        model = DoorOpenAggregate

    group_key = factory.fuzzy.FuzzyDateTime(
        (timezone.now() - timedelta(days=30)).replace(
            minute=0, second=0, microsecond=0
        )
    )
    group_by = AggregateGroup.HOUR
    max_timestamp = factory.LazyAttribute(
        lambda o: o.group_key + timedelta(minutes=59)
    )

    organization = None
    zone = None
    camera = None

    open_time_duration_s = 0
    close_time_duration_s = 0
    partially_open_time_duration_s = 0


class StateFactory(DjangoModelFactory):
    """State factory."""

    class Meta:
        """Meta class used for factory configuration."""

        model = State

    timestamp = timezone.now()
    end_timestamp = timezone.now()
    run_uuid = ""

    camera_uuid = ""
    organization = ""
    location = ""
    zone = ""
    camera_name = ""
    actor_id = ""
    actor_category = factory.fuzzy.FuzzyChoice(enum_values(ActorCategory))

    # Person fields
    person_activity_type = factory.fuzzy.FuzzyChoice(enum_values(ActivityType))
    person_posture_type = factory.fuzzy.FuzzyChoice(enum_values(PostureType))
    person_lift_type = factory.fuzzy.FuzzyChoice(enum_values(PostureType))
    person_reach_type = factory.fuzzy.FuzzyChoice(enum_values(PostureType))
    person_is_wearing_safety_vest = None
    person_is_wearing_hard_hat = None
    person_is_carrying_object = None
    person_is_associated = None
    person_in_no_ped_zone = None

    # PIT fields
    pit_is_stationary = None
    pit_in_driving_area = None
    pit_is_associated = None

    # Door fields
    door_is_open = None

    # Motion fields
    motion_zone_is_in_motion = False


class EventFactory(DjangoModelFactory):
    """Event factory."""

    class Meta:
        """Meta class used for factory configuration."""

        model = Event

    timestamp = timezone.now()
    camera_uuid = ""
    organization = ""
    location = ""
    zone = ""
    camera_name = ""
    subject_id = 0
    object_id = 0
    event_type = factory.fuzzy.FuzzyChoice(enum_values(EventType))
    end_timestamp = timezone.now()
    run_uuid = ""
    x_velocity_pixel_per_sec = 0.0
    y_velocity_pixel_per_sec = 0.0
    normalized_speed = 0.0


class ShareLinkFactory(
    DjangoModelFactory
):  # pylint: disable=too-few-public-methods
    """ShareLink factory creates an instance of share links of an incident for use in testing."""

    class Meta:  # pylint: disable=too-few-public-methods
        """Meta is metainfo on this class"""

        model = ShareLink

    token = factory.LazyFunction(uuid4)
    expires_at = timezone.now() + timedelta(days=3)
    shared_by = factory.SubFactory(UserFactory)
    visits = 0

    incident = factory.SubFactory(IncidentFactory)


class CameraIncidentTypeFactory(DjangoModelFactory):
    """CameraIncidentTypeFactory factory."""

    class Meta:
        """Factory meta class."""

        model = CameraIncidentType

    camera = factory.SubFactory(CameraFactory)
    incident_type = factory.SubFactory(IncidentTypeFactory)
    enabled = True


class PerceivedActorStateDurationAggregateFactory(DjangoModelFactory):
    class Meta:
        model = PerceivedActorStateDurationAggregate

    time_bucket_width = TimeBucketWidth.HOUR.value
    duration = factory.LazyAttribute(
        lambda _: timedelta(minutes=random.randrange(0, 86400))
    )
