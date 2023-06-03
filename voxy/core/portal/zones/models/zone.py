#
# Copyright 2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from django.apps import apps
from django.contrib import admin
from django.contrib.auth.models import User
from django.db import models
from django.db.models.query import QuerySet

from core.portal.api.models.organization import Organization
from core.portal.lib.models.base import Model
from core.portal.zones.enums import ZoneType

if TYPE_CHECKING:
    from core.portal.api.models.incident_type import OrganizationIncidentType
    from core.portal.compliance.models.zone_compliance_type import (
        ZoneComplianceType,
    )


class Zone(Model):
    class Meta:
        app_label = "zones"
        db_table = "zones"

        constraints = [
            models.UniqueConstraint(
                fields=["organization", "parent_zone", "key"],
                name="unique_key_per_org_and_parent",
            )
        ]

    def __str__(self) -> str:
        return self.name

    anonymous_key = models.UUIDField(unique=True, null=True, blank=True)
    parent_zone = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        max_length=250,
    )

    active = models.BooleanField(
        null=False,
        default=True,
        help_text=(
            "True if Voxel production systems should be running at"
            + " this site, otherwise False.",
        ),
    )
    key = models.CharField(max_length=50, null=False, blank=False)

    name = models.CharField(
        max_length=250,
        null=False,
        blank=False,
    )

    timezone = models.CharField(
        help_text="Optional field specifying the zone timezone, if null then inherit the timezone from parent zone or organization",
        max_length=50,
        choices=Model.timezones(),
        null=True,
        blank=True,
    )

    zone_type = models.CharField(
        max_length=10,
        choices=ZoneType.choices,
        default=ZoneType.SITE,
        null=False,
        blank=False,
    )

    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        null=False,
        blank=False,
        related_name="zones",
    )

    users = models.ManyToManyField(
        User, through="ZoneUser", related_name="zones", blank=True
    )

    config = models.JSONField(null=True, blank=True)

    is_high_priority = models.BooleanField(default=False)

    @property
    def tzinfo(self) -> ZoneInfo:
        if self.timezone:
            return ZoneInfo(self.timezone)
        if self.parent_zone:
            return self.parent_zone.tzinfo
        return self.organization.tzinfo

    @property
    def enabled_incident_types(self) -> QuerySet["OrganizationIncidentType"]:
        """Get enabled incident types.

        Returns:
            QuerySet[OrganizationIncidentType]: enabled incident type queryset
        """
        # TODO(PRO-301): set up proper zone-level incident types
        camera_incident_type_model = apps.get_model("api.CameraIncidentType")
        enabled_incident_type_ids = camera_incident_type_model.objects.filter(
            camera__zone_id=self.id,
            enabled=True,
        ).values_list("incident_type_id", flat=True)
        return self.organization.enabled_incident_types.filter(
            incident_type_id__in=enabled_incident_type_ids
        )

    @property
    def active_users(self) -> QuerySet[User]:
        return self.users.filter(is_active=True)

    @property
    def assignable_users(self) -> QuerySet[User]:
        """
        Get all the assignable users of the site.

        Returns:
            QuerySet[User]:  QuerySet that resolves to the list of users in the
            site where is_assignable=True
        """
        users = self.active_users.filter(
            zone_users__in=self.zone_users.filter(is_assignable=True)
        )
        return users

    @property
    def enabled_zone_compliance_types(self) -> "QuerySet[ZoneComplianceType]":
        """Enabled zone compliance types.

        NOTE: this returns instances of ZoneComplianceType, not ComplianceType,
              but we prefetch the compliance_type related field so callers
              can access the underlying ComplianceType instance without
              triggering N+1 query scenarios.

        Returns:
            QuerySet[ZoneComplianceType]: enabled compliance types for a zone
        """
        return self.zone_compliance_types.filter(
            enabled=True
        ).prefetch_related("compliance_type")


class ZoneUser(Model):
    class Meta:
        unique_together = (
            "zone",
            "user",
        )

    zone = models.ForeignKey(
        Zone,
        on_delete=models.CASCADE,
        related_name="zone_users",
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="zone_users",
    )

    is_assignable = models.BooleanField(default=True, null=False)


class ZoneUserInline(admin.TabularInline):
    model = ZoneUser
    extra = 1


class ZoneAdmin(admin.ModelAdmin):
    filter_horizontal = ("users",)
    inlines = (ZoneUserInline,)
