from zoneinfo import ZoneInfo

from django.apps import apps
from django.contrib import admin
from django.contrib.auth.models import User
from django.db import models
from django.db.models import Subquery

from core.portal.api.models.comment import Comment
from core.portal.lib.models.base import Model
from core.portal.zones.enums import ZoneType


class Organization(Model):
    """Represents a customer/account/organization.

    The idea is organizations may be further divided into buildings, sites,
    sub-organizations, etc. but this is the top-level identifier.
    """

    anonymous_key = models.UUIDField(unique=True, null=True, blank=True)

    name = models.CharField(max_length=250, null=False, blank=False)
    key = models.CharField(max_length=50, null=False, blank=False, unique=True)
    users = models.ManyToManyField(
        User, related_name="organizations", blank=True
    )
    is_sandbox = models.BooleanField(default=False)
    timezone = models.CharField(
        max_length=50, choices=Model.timezones(), default="US/Pacific"
    )

    def __str__(self) -> str:
        """Return the string representation of an organization."""
        return self.name

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def sites(self):
        return self.zones.filter(zone_type=ZoneType.SITE)

    @property
    def enabled_incident_types(self):
        """Enabled incident types for organization

        Returns:
            QuerySet["CameraIncidentType"]: enabled camera incident types
        """
        camera_incident_type_model = apps.get_model("api.CameraIncidentType")
        enabled_incident_type_ids = Subquery(
            camera_incident_type_model.objects.filter(
                camera__organization_id=self.id,
                enabled=True,
            ).values_list("incident_type_id", flat=True)
        )
        return self.organization_incident_types.filter(
            incident_type_id__in=enabled_incident_type_ids
        ).prefetch_related("incident_type")

    @property
    def comments(self):
        return Comment.objects.filter(
            incident__organization=self,
            owner__in=self.users.all(),
        )

    @property
    def active_users(self):
        return self.users.filter(
            is_active=True,
        )


class OrganizationAdmin(admin.ModelAdmin):
    filter_horizontal = ("users",)
