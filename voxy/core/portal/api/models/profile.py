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

from datetime import datetime, timezone
from typing import Optional, Set, cast

from django.contrib import admin
from django.contrib.auth.models import User
from django.db import models
from django.db.models.query import QuerySet

from core.portal.accounts.models.user_role import UserRole
from core.portal.api.models.incident import Incident
from core.portal.api.models.list import STARRED_LIST_NAME, List
from core.portal.api.models.organization import Organization
from core.portal.lib.models.base import Model
from core.portal.lib.utils.date_utils import convert_tz
from core.portal.zones.models.zone import Zone


class ProfileAdmin(admin.ModelAdmin):
    search_fields = ["owner__email"]


class Profile(Model):
    """Used to attach fields to the User model."""

    class Meta:
        permissions = (
            ("can_access_incident_feed", "Can access incident feed."),
            ("can_access_analytics_page", "Can access analytics page."),
            ("can_access_live_feed", "Can access live feed."),
            ("can_review_incidents", "Can review incidents."),
            (
                "manage_incident_review_process",
                "Can manage the incident review process.",
            ),
        )

    owner = models.OneToOneField(User, on_delete=models.CASCADE)
    organization = models.ForeignKey(
        Organization, null=True, on_delete=models.RESTRICT
    )
    site = models.ForeignKey(
        Zone,
        null=True,
        blank=True,
        # RESTRICT: require all users be assigned to a different site before
        # a site can be deleted
        on_delete=models.RESTRICT,
    )
    data = models.JSONField(null=True, blank=True)
    timezone = models.CharField(
        max_length=50, choices=Model.timezones(), default="US/Pacific"
    )

    def __str__(self):
        return self.owner.email

    @property
    def current_organization(self) -> Optional[Organization]:
        # Admins can switch between organizations
        if self.owner.is_superuser:
            if self.organization:
                return self.organization
            if self.data:
                key = self.data.get("current_organization_key")
                if key:
                    return Organization.objects.get(key=key)
            # Assign default organization
            org = Organization.objects.all().first()
            if org:
                self.organization = org
                self.save()
                return cast(Organization, org)
            return None
        return self.owner.organizations.first()

    def avatar_url(self) -> str:
        if self.data:
            return self.data.get("avatarUrl", "")
        return ""

    @property
    def starred_list(self) -> List:
        try:
            return List.objects.get(owner=self.owner, is_starred_list=True)
        except List.DoesNotExist:
            # TODO: remove init logic after we're confident this list already exists
            return List.objects.create(
                owner=self.owner, name=STARRED_LIST_NAME, is_starred_list=True
            )

    @property
    def starred_incidents(self) -> List:
        """Bookmarked (starred) incidents scoped to the current zone."""
        return self.starred_list.incidents.for_user(self.owner)

    @property
    def incidents_assigned_to_me(self) -> QuerySet[Incident]:
        """Incidents assigned by this user - scoped to current zone."""
        # pylint: disable=protected-access
        return self.owner._incidents_assigned_to_me.for_user(self.owner)

    @property
    def incidents_assigned_by_me(self) -> QuerySet[Incident]:
        """Incidents assigned to this user - scoped to current zone."""
        # pylint: disable=protected-access
        return self.owner._incidents_assigned_by_me.for_user(self.owner)

    @property
    def local_time(self):
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        user_tz = self.timezone if self.timezone else "US/Pacific"
        return convert_tz(utc_now, "UTC", user_tz)

    @property
    def receive_daily_summary_emails(self):
        return (self.data or {}).get("receive_daily_summary_emails", False)

    @property
    def permissions(self) -> Set[str]:
        return set(
            UserRole.objects.filter(
                user_id=self.owner.id,
                removed_at__isnull=True,
                role__role_permissions__removed_at__isnull=True,
            ).values_list(
                "role__role_permissions__permission_key",
                flat=True,
            )
        )
