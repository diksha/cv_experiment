from datetime import datetime, timezone
from typing import List, Optional

from django.conf import settings
from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.postgres.indexes import BTreeIndex
from django.db import models
from django.db.models import Exists, F, OuterRef, Q, Subquery
from django.db.models.query import QuerySet
from django_cte import CTEManager, CTEQuerySet

from core.portal.api.models.incident_filters import apply_filter
from core.portal.api.models.incident_type import IncidentType
from core.portal.api.models.organization import Organization
from core.portal.devices.models.camera import Camera, CameraConfigNew
from core.portal.incidents.enums import (
    CooldownSource,
    IncidentPriority,
    ReviewStatus,
    ScenarioType,
)
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.incidents.types import Filter
from core.portal.lib.models.base import Model
from core.portal.lib.utils.signed_url_manager import signed_url_manager
from core.portal.zones.models.zone import Zone
from core.utils.aws_utils import copy_object


class IncidentQuerySet(CTEQuerySet):
    def from_timestamp(
        self, timestamp: Optional[datetime]
    ) -> "IncidentQuerySet":
        if timestamp:
            return self.filter(timestamp__gte=timestamp)
        return self

    def to_timestamp(
        self, timestamp: Optional[datetime]
    ) -> "IncidentQuerySet":
        if timestamp:
            return self.filter(timestamp__lte=timestamp)
        return self

    def from_last_feedback_submission_timestamp(
        self, timestamp: Optional[datetime]
    ) -> "IncidentQuerySet":
        """Generates expression to filter for all incidents who have
        a feedback submissions after the given timestamp

        Args:
            timestamp (Optional[datetime]): timestamp

        Returns:
            IncidentQuerySet: Queryset for incidents who have a
            feedback submission after the timestamp
        """
        if timestamp:
            return self.filter(
                last_feedback_submission_timestamp__gte=timestamp
            )
        return self

    def to_last_feedback_submission_timestamp(
        self, timestamp: Optional[datetime]
    ) -> "IncidentQuerySet":
        """Generates expression to filter for all incidents who have
        a feedback submissions before the given timestamp

        Args:
            timestamp (Optional[datetime]): timestamp

        Returns:
            IncidentQuerySet: Queryset for incidents who have a
            feedback submission before the timestamp
        """
        if timestamp:
            return self.filter(
                last_feedback_submission_timestamp__lte=timestamp
            )
        return self

    def for_organization(
        self,
        organization: Organization,
        zones: List[Zone],
        superuser: bool = False,
    ) -> "IncidentQuerySet":
        """Filters queryset to incidents visible to a specific organization."""

        if not list(filter(None, zones)):
            raise RuntimeError("All queries must be scoped to a site/zone")

        enabled_incident_type_keys = {
            t.key for t in organization.enabled_incident_types
        }

        queryset = (
            self.prefetch_related("camera").filter(
                # Only show enabled incident types from current organization
                visible_to_customers=True,
                organization=organization,
                zone_id__in=zones,
                incident_type__key__in=enabled_incident_type_keys,
                # Hide soft-deleted incidents
                deleted_at__isnull=True,
            )
            # Exclude sub-incidents
            .exclude(
                Q(
                    data__has_keys=[
                        "start_frame_relative_ms",
                        "incident_group_start_time_ms",
                    ]
                ),
                # does NOT equal None (tilde negates Q criteria)
                ~Q(data__incident_group_start_time_ms=None),
                Q(
                    data__start_frame_relative_ms__gt=F(
                        "data__incident_group_start_time_ms"
                    )
                ),
            )
        )

        return queryset

    def for_site(self, site: Zone) -> "IncidentQuerySet":
        """Filteres queryset to incidents visible to a specific site.

        Args:
            site (Zone): site

        Returns:
            IncidentQuerySet: filtered queryset
        """
        return self.for_organization(
            site.organization,
            [site],
        )

    def for_user(self, user: User) -> "IncidentQuerySet":
        """Filters queryset to incidents visible to a specific user."""
        return self.for_organization(
            user.profile.current_organization,
            [user.profile.site],
            superuser=user.is_superuser,
        )

    def with_bookmarked_flag(self, user: User) -> "IncidentQuerySet":
        """Annotates queryset with `bookmarked` boolean."""
        bookmarked_incidents = user.profile.starred_list.incidents.filter(
            id=OuterRef("id")
        )
        return self.annotate(
            bookmarked=Exists(Subquery(bookmarked_incidents)),
        )

    def apply_filters(
        self, filters: List[Filter], current_user: User
    ) -> QuerySet["Incident"]:
        """Applies all provided filters to the queryset."""
        queryset: QuerySet[Incident] = self
        for filter_data in filters:
            queryset = apply_filter(queryset, filter_data, current_user)
        return queryset.distinct()


class DefaultManager(CTEManager):
    """Filtered manager which only includes customer-facing incidents."""

    def get_queryset(self) -> IncidentQuerySet:
        """Get the manager queryset.

        Returns:
            IncidentQuerySet: filtered incident queryset
        """
        # TODO: use visible_to_customers field to filter this queryset
        return IncidentQuerySet(self.model, using=self._db).filter(
            # Exclude experimental incidents
            experimental=False,
            # Exclude cooldown incidents
            cooldown_source__isnull=True,
        )


class ReviewableIncidentsManager(CTEManager):
    """Filtered manager which only includes reviewable incidents."""

    def get_queryset(self) -> IncidentQuerySet:
        """Get the manager queryset.

        Returns:
            IncidentQuerySet: filtered incident queryset
        """
        return IncidentQuerySet(self.model, using=self._db).filter(
            review_status__in=(
                ReviewStatus.NEEDS_REVIEW,
                ReviewStatus.VALID_AND_NEEDS_REVIEW,
            ),
        )


class AggregableIncidentsManager(CTEManager):
    """Filtered manager which only includes aggregable incidents.

    aggregable (adjective): able to be aggregated.
    """

    def get_queryset(self) -> IncidentQuerySet:
        """Get the manager queryset.

        Returns:
            IncidentQuerySet: filtered incident queryset
        """
        return IncidentQuerySet(self.model, using=self._db).filter(
            # Exclude experimental incidents
            experimental=False,
            # Only include valid incidents
            review_status=ReviewStatus.VALID,
        )


class ExperimentalManager(CTEManager):
    """Filtered manager which only includes experimental incidents."""

    def get_queryset(self) -> IncidentQuerySet:
        """Get the manager queryset.

        Returns:
            IncidentQuerySet: filtered incident queryset
        """
        return IncidentQuerySet(self.model, using=self._db).filter(
            # Only include experimental incidents
            experimental=True
        )


class RawManager(CTEManager):
    """Unfiltered access to all objects, please only use for admin purposes."""

    def get_queryset(self) -> IncidentQuerySet:
        """Get the manager queryset.

        Returns:
            IncidentQuerySet: unfiltered incident queryset
        """
        return IncidentQuerySet(self.model, using=self._db)


# trunk-ignore(pylint/R0904): aggregate object needs many public classes
class Incident(Model):
    class Meta:
        indexes = [
            models.Index(
                fields=[
                    "-timestamp",
                ]
            ),
            # TODO: This index could get very big, consider "archiving" in the future
            BTreeIndex(
                fields=["-last_feedback_submission_timestamp"],
                name="last_feedback_submis_nonnull_idx",
                condition=Q(last_feedback_submission_timestamp__isnull=False),
            ),
            # Creating a composite partial index
            models.Index(
                fields=["review_status"],
                name="review_status_partial_idx",
                condition=Q(
                    review_status__in=[
                        ReviewStatus.NEEDS_REVIEW,
                        ReviewStatus.VALID_AND_NEEDS_REVIEW,
                    ],
                ),
            ),
        ]

    class Priority(models.TextChoices):
        LOW = (IncidentPriority.LOW.value, "Low Priority")
        MEDIUM = (IncidentPriority.MEDIUM.value, "Medium Priority")
        HIGH = (IncidentPriority.HIGH.value, "High Priority")

    class Status(models.TextChoices):
        OPEN = ("open", "Open")
        IN_PROGRESS = ("in_progress", "In Progress")
        RESOLVED = ("resolved", "Resolved")

    # Custom model managers
    objects = DefaultManager.from_queryset(IncidentQuerySet)()
    objects_experimental = ExperimentalManager.from_queryset(
        IncidentQuerySet
    )()
    objects_raw = RawManager.from_queryset(IncidentQuerySet)()
    reviewable_objects = ReviewableIncidentsManager.from_queryset(
        IncidentQuerySet
    )()
    aggregable_objects = AggregableIncidentsManager.from_queryset(
        IncidentQuerySet
    )()

    # TODO: mark as required after all existing data has been populated
    uuid = models.UUIDField(unique=True, null=True, blank=True)
    title = models.CharField(max_length=100)
    timestamp = models.DateTimeField()
    last_feedback_submission_timestamp = models.DateTimeField(
        null=True, blank=True
    )
    data = models.JSONField(null=True, blank=True)
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        related_name="incidents",
    )
    camera = models.ForeignKey(
        Camera,
        on_delete=models.RESTRICT,
        null=True,
        blank=True,
        related_name="incidents",
    )
    zone = models.ForeignKey(
        Zone,
        on_delete=models.RESTRICT,
        null=True,
        blank=True,
        related_name="incidents",
    )
    incident_type = models.ForeignKey(
        IncidentType, on_delete=models.CASCADE, blank=True, null=True
    )
    priority = models.CharField(
        max_length=20, choices=Priority.choices, default=Priority.MEDIUM
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.OPEN
    )
    highlighted = models.BooleanField(
        # TODO: mark null=False, default=False once existing data has been backfilled
        blank=True,
        null=True,
        help_text=(
            "Highlighted incidents are especially valuable or interesting to"
            + " customers and get special treatment within the application,"
            + " such as being included in daily summary emails or getting"
            + " displayed at the top of the dashboard."
        ),
    )
    experimental = models.BooleanField(null=False, default=False)
    assigned_by = models.ManyToManyField(
        User,
        through="UserIncident",
        through_fields=("incident", "assigned_by"),
        # NOTE: This relation should only be consumed via the profile model
        #       to ensure additional required filters are applied.
        related_name="_incidents_assigned_by_me",
    )
    assigned_to = models.ManyToManyField(
        User,
        through="UserIncident",
        through_fields=("incident", "assignee"),
        # NOTE: This relation should only be consumed via the profile model
        #       to ensure additional required filters are applied.
        related_name="_incidents_assigned_to_me",
    )
    alerted = models.BooleanField(
        null=False,
        default=False,
        help_text="True if any alerts have been sent for this incident, otherwise False",
    )
    cooldown_source = models.PositiveSmallIntegerField(
        null=True, blank=True, choices=CooldownSource.choices
    )
    visible_to_customers = models.BooleanField(null=False, default=False)

    # Feedback denormalization
    valid_feedback_count = models.PositiveSmallIntegerField(
        null=False, default=0
    )
    invalid_feedback_count = models.PositiveSmallIntegerField(
        null=False, default=0
    )
    unsure_feedback_count = models.PositiveSmallIntegerField(
        null=False, default=0
    )
    corrupt_feedback_count = models.PositiveSmallIntegerField(
        null=False, default=0
    )
    review_level: models.CharField = models.CharField(
        max_length=20, choices=ReviewLevel.choices, default=ReviewLevel.RED
    )
    review_status = models.PositiveSmallIntegerField(
        null=True, choices=ReviewStatus.choices
    )

    def __str__(self):
        return f"{self.uuid} - {self.title}"

    # TODO: remove this property when self.uuid is marked as required
    @property
    def uuid_wrapper(self) -> str:
        return self.uuid or (self.data or {}).get("uuid", "")

    @property
    def camera_uuid(self) -> str:
        return (self.data or {}).get("camera_uuid", "")

    @property
    def camera_config(self) -> CameraConfigNew:
        return CameraConfigNew.objects.get(
            camera=self.camera,
            version=(self.data or {}).get("camera_config_version", 1),
        )

    @property
    def end_timestamp(self) -> Optional[datetime]:
        """
        Return the end timestamp of the event in UTC timezone.

        Returns:
            Optional[datetime]: The end timestamp of the event, in UTC timezone,
            or None if the `end_frame_relative_ms` attribute is not found.
        """
        end_frame_relative_ms = (self.data or {}).get("end_frame_relative_ms")

        if not end_frame_relative_ms:
            return None

        end_frame_relative_s = float(end_frame_relative_ms) / 1000.0
        return datetime.fromtimestamp(end_frame_relative_s, tz=timezone.utc)

    @property
    def start_timestamp(self) -> Optional[datetime]:
        """
        Return the start timestamp of the event in UTC timezone.

        Returns:
            Optional[datetime]: The start timestamp of the event, in UTC timezone,
            or None if the `start_frame_relative_ms` attribute is not found.
        """
        start_frame_relative_ms = (self.data or {}).get(
            "start_frame_relative_ms"
        )

        if not start_frame_relative_ms:
            return None

        start_frame_relative_s = float(start_frame_relative_ms) / 1000.0
        return datetime.fromtimestamp(start_frame_relative_s, tz=timezone.utc)

    @property
    def duration(self) -> Optional[int]:
        """
        Return the duration in seconds of the event.

        Returns:
            Optional[timedelta]: The duration of the event in seconds, as a timedelta object,
            or None if either the start or end timestamp is missing.
        """
        start_timestamp = self.start_timestamp
        end_timestamp = self.end_timestamp

        if not start_timestamp or not end_timestamp:
            return None

        delta = end_timestamp - start_timestamp
        return int(delta.total_seconds())

    @property
    def video_thumbnail_s3_path(self) -> Optional[str]:
        """Property that defines potential s3 video thumbnail path.

        Returns:
            Optional[str]: video thumbnail path
        """
        return (self.data or {}).get("video_thumbnail_s3_path")

    @property
    def thumbnail_url(self):
        if self.video_thumbnail_s3_path:
            return signed_url_manager.get_signed_url(
                s3_path=self.video_thumbnail_s3_path,
            )
        return ""

    @property
    def thumbnail_url_mrap(self):
        """Thumbnail url multi-region access

        Returns:
            str: thumbnail path
        """
        if self.video_thumbnail_s3_path:
            return signed_url_manager.get_signed_url(
                s3_path=self.video_thumbnail_s3_path,
                enable_multi_region_access=True,
            )
        return ""

    @property
    def video_s3_path(self) -> Optional[str]:
        """Property that defines the potential s3 video path
        Returns:
            Optional[str]: the video path
        """
        return (self.data or {}).get("video_s3_path")

    @property
    def original_video_s3_path(self) -> Optional[str]:
        """Path of video with original encoding.
        Returns:
            Optional[str]: path of video if set
        """
        return (self.data or {}).get("original_video_s3_path")

    @property
    def annotations_s3_path(self) -> Optional[str]:
        """Property that defines the potential annotations s3 path
        Returns:
            Optional[str]: the path to the annotations in s3
        """
        return (self.data or {}).get("annotations_s3_path")

    @property
    def video_annotated_s3_path(self) -> Optional[str]:
        """Property that defines the potential video annotations s3 path

        Returns:
            Optional[str]: the path to the annotated video in s3
        """
        return (self.data or {}).get("video_annotated_s3_path")

    @property
    def video_url(self) -> str:
        """
        Returns a signed URL for the video stored in an S3 bucket.

        If the `video_s3_path` attribute is set, this method generates a signed URL
        for the video file stored in the corresponding S3 bucket using a signed_url_manager
        instance. Otherwise, an empty string is returned.

        Returns:
            str: A signed URL for the video file stored in the S3 bucket, or an empty
            string if `video_s3_path` is not set.
        """
        if self.video_s3_path:
            return signed_url_manager.get_signed_url(
                s3_path=self.video_s3_path,
            )

        return ""

    @property
    def video_url_mrap(self) -> str:
        """Video url multi-region path

        Returns:
            str: returns video url s3 path
        """
        if self.video_s3_path:
            return signed_url_manager.get_signed_url(
                s3_path=self.video_s3_path,
                enable_multi_region_access=True,
            )

        return ""

    @property
    def annotations_url(self) -> str:
        if self.annotations_s3_path:
            return signed_url_manager.get_signed_url(
                s3_path=self.annotations_s3_path,
            )
        return ""

    @property
    def annotations_url_mrap(self) -> str:
        """Annotations path with multi-region access

        Returns:
            str: returns annotations s3 path
        """
        if self.annotations_s3_path:
            return signed_url_manager.get_signed_url(
                s3_path=self.annotations_s3_path,
                enable_multi_region_access=True,
            )
        return ""

    @property
    def actor_ids(self) -> List[str]:
        """Return IDs of actors involved in this incident.

        data.actor_ids is a string containing a comma separated list of IDs
        so we need to split it into a list of strings:

        "1,2,3"   -> ["1", "2", "3"]
        "1,a,foo" -> ["1", "a", "foo"]
        " a, 1, " -> ["a", "1"]
        ",1,2,"   -> ["1", "2"]
        "1"       -> ["1"]

        Returns:
            List[str]: actor IDs
        """
        id_string = (self.data or {}).get("actor_ids", "")
        id_string = id_string.strip(" ,")  # strip both whitespace and commas
        return [x.strip() for x in id_string.split(",") if x]

    @property
    def docker_image_tag(self) -> Optional[str]:
        tag = (self.data or {}).get("docker_image_tag")
        if tag:
            return tag.split("_")[-1]
        return None

    @property
    def incident_version(self) -> Optional[str]:
        return (self.data or {}).get("incident_version")

    @property
    def is_cooldown(self) -> bool:
        """Flag representing whether this incident is a cooldown incident.

        Returns:
            bool: true if cooldown incident, otherwise false
        """
        return self.cooldown_source is not None

    def sub_incidents(self):
        """Returns incidents which are "sub incidents" of this instance."""
        if (
            self.data
            and self.data.get("start_frame_relative_ms")
            and self.data.get("incident_group_start_time_ms")
            # Make sure this instance is the "base" incident
            and self.data.get("start_frame_relative_ms")
            == self.data.get("incident_group_start_time_ms")
        ):
            return (
                Incident.objects.filter(
                    Q(
                        data__has_keys=[
                            "start_frame_relative_ms",
                            "incident_group_start_time_ms",
                        ]
                    ),
                    # does NOT equal None (tilde negates Q criteria)
                    ~Q(data__incident_group_start_time_ms=None),
                    Q(
                        data__incident_group_start_time_ms=self.data.get(
                            "incident_group_start_time_ms"
                        )
                    ),
                )
                # Exclude self
                .exclude(id=self.id)
            )
        return Incident.objects.none()

    def create_scenario(self, scenario_type: ScenarioType):
        """Copies incident video to scenarios directory in S3.

        Only create scenarios in prod until we have pre-prod scenario buckets.

        Args:
            scenario_type (ScenarioType): Scenario type
        """
        video_s3_path_to_copy = (
            self.original_video_s3_path
            if self.original_video_s3_path
            else self.video_s3_path
        )

        bucket = "voxel-raw-logs"
        # trunk-ignore(pylint/C0301): line too long is ok for uri
        scenario_s3_uri = f"s3://{bucket}/{self.camera_uuid}/scenarios/{self.incident_type.key}/{scenario_type}/{self.uuid}.mp4"

        if settings.PRODUCTION and video_s3_path_to_copy:
            copy_object(video_s3_path_to_copy, scenario_s3_uri)
            self.data["scenario_s3_uri"] = scenario_s3_uri
            self.save()

    @classmethod
    def is_experimental_version(cls, version: str) -> bool:
        """Determines if a given version string is considered experimental.

        Args:
            version (str): incident version string

        Returns:
            bool: flag representing whether incident is considered experimental
        """
        return version is not None and version.lower().startswith(
            "experimental"
        )


class IncidentAdmin(admin.ModelAdmin):
    def get_queryset(self, *args: str) -> "QuerySet[Incident]":
        """Use raw model manager on admin site."""
        del args
        qs = self.model.objects_raw.get_queryset()
        ordering = self.ordering or ()
        if ordering:
            qs = qs.order_by(*ordering)
        return qs


class UserIncident(Model):

    organization = models.ForeignKey(
        Organization, on_delete=models.CASCADE, blank=True, null=True
    )
    incident = models.ForeignKey(
        Incident, on_delete=models.CASCADE, related_name="user_incidents"
    )
    assigned_by = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        related_name="assigned_by",
    )
    assignee = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        related_name="assignee",
    )
    note = models.CharField(max_length=1000, null=True, blank=True)

    def __str__(self):
        return f"{self.incident}"

    class Meta:
        unique_together = (
            "incident",
            "assignee",
        )


class UserIncidentAdmin(admin.ModelAdmin):
    list_display = ("organization", "incident", "assigned_by", "assignee")
