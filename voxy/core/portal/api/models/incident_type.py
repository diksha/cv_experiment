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
from typing import Any

from django.db import models

from core.portal.devices.models.camera import Camera
from core.portal.incidents.enums import IncidentCategory
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.lib.models.base import Model
from core.portal.perceived_data.enums import (
    PerceivedActorStateDurationCategory,
)
from core.portal.zones.models.zone import Zone

# TODO(PORTAL-166): get mypy working with model fields


class IncidentType(Model):
    class Category(models.TextChoices):
        VEHICLE = (IncidentCategory.VEHICLE.value, "VEHICLE")
        ENVIRONMENT = (IncidentCategory.ENVIRONMENT.value, "ENVIRONMENT")
        PEOPLE = (IncidentCategory.PEOPLE.value, "PEOPLE")

    @classmethod
    def cache_key(cls, incident_type_id: int) -> str:
        return f"orgIncidentType/{incident_type_id}"

    organizations = models.ManyToManyField(
        "Organization",
        through="OrganizationIncidentType",
        related_name="incident_types",
    )
    cameras = models.ManyToManyField(
        Camera,
        through="CameraIncidentType",
        related_name="incident_types",
    )

    key = models.CharField(
        max_length=100, null=False, blank=False, unique=True
    )
    name = models.CharField(
        max_length=100, null=False, blank=False, unique=True
    )
    description = models.TextField(blank=True, null=True)

    background_color = models.CharField(max_length=7)
    category = models.CharField(
        max_length=25, choices=Category.choices, null=True, blank=True
    )
    perceived_event_rate_denominator_category = models.PositiveSmallIntegerField(
        blank=True,
        null=True,
        choices=PerceivedActorStateDurationCategory.choices,
        help_text=(
            "The perceived actor state duration category used as the denominator"
            + " in perceived event rate calculations for this incident type."
        ),
    )
    # TODO: deprecate the `value` field in favor of `key`
    value = models.CharField(
        max_length=100, null=False, blank=False, unique=True
    )

    def __str__(self) -> str:
        return self.key

    def save(self, *args: Any, **kwargs: Any) -> None:
        self.value = "_".join(self.value.split()).upper()
        super().save(*args, **kwargs)


class OrganizationIncidentType(Model):
    class Meta:
        """Django model meta class."""

        unique_together = (
            "incident_type",
            "organization",
        )

    incident_type = models.ForeignKey(
        IncidentType,
        on_delete=models.CASCADE,
        related_name="organization_incident_types",
    )
    organization = models.ForeignKey(
        "Organization",
        on_delete=models.CASCADE,
        related_name="organization_incident_types",
    )
    enabled = models.BooleanField(default=True)
    # Optional organization-specific name override for this incident type
    name_override = models.CharField(max_length=100, null=True, blank=True)
    review_level: models.CharField = models.CharField(
        max_length=20, choices=ReviewLevel.choices, default=ReviewLevel.RED
    )

    @classmethod
    def cache_key(cls, incident_type_id: int, organization_id: int) -> str:
        incident_type_key = IncidentType.cache_key(incident_type_id)
        return f"{incident_type_key}/{organization_id}"

    def __str__(self):
        return (
            f"{self.incident_type} - {self.organization} - {self.review_level}"
        )

    @property
    def key(self):
        return self.incident_type.key

    @property
    def category(self) -> str:
        return self.incident_type.category

    @property
    def name(self):
        return self.name_override or self.incident_type.name

    @property
    def background_color(self):
        return self.incident_type.background_color


class CameraIncidentType(Model):
    class Meta:
        """Django model meta class."""

        db_table = "camera_incident_type"
        unique_together = (
            "incident_type",
            "camera",
        )

    incident_type = models.ForeignKey(
        IncidentType,
        on_delete=models.CASCADE,
        related_name="camera_incident_types",
    )
    camera = models.ForeignKey(
        Camera,
        on_delete=models.CASCADE,
        related_name="camera_incident_types",
    )
    enabled = models.BooleanField(default=True)
    review_level: models.CharField = models.CharField(
        max_length=20, choices=ReviewLevel.choices, default=ReviewLevel.RED
    )

    @classmethod
    def cache_key(cls, incident_type_id: int, camera_id: int) -> str:
        """Returns cache key.

        Args:
            incident_type_id (int): incident type id to use
            camera_id (int): camera id to use

        Returns:
            str: cache key
        """
        incident_type_key = IncidentType.cache_key(incident_type_id)
        return f"{incident_type_key}/{camera_id}"

    def __str__(self) -> str:
        """Returns string representation of class

        Returns:
            str: string representation of class
        """
        return f"{self.incident_type} - {self.camera} - {self.review_level}"

    @property
    def key(self) -> str:
        """Returns key of incident type

        Returns:
            str: key of incident type
        """
        return self.incident_type.key

    @property
    def category(self) -> str:
        """Returns category of incident type

        Returns:
            str: category of incident type
        """
        return self.incident_type.category

    @property
    def name(self) -> str:
        """Returns the name of incident type

        Returns:
            str: name of incident type
        """
        return self.incident_type.name

    @property
    def background_color(self) -> str:
        """Returns background color of incident type

        Returns:
            str: background color
        """
        return self.incident_type.background_color

    @property
    def description(self) -> str:
        """Get the description of the incident.

        If a description_override is set, return that description. Otherwise, return
        the description of the incident_type associated with this incident.

        Returns:
            str: The description of the incident.
        """
        site_incident_type = self.incident_type.site_incident_types.filter(
            site=self.camera.zone
        ).first()

        return (
            site_incident_type.description_override
            if site_incident_type
            and site_incident_type.description_override is not None
            else self.incident_type.description
        )


class SiteIncidentType(Model):
    class Meta:
        """Django model meta class."""

        db_table = "site_incident_type"
        unique_together = (
            "incident_type",
            "site",
        )

    incident_type = models.ForeignKey(
        IncidentType,
        on_delete=models.CASCADE,
        related_name="site_incident_types",
    )
    site = models.ForeignKey(
        Zone, on_delete=models.CASCADE, related_name="site_incident_types"
    )
    enabled = models.BooleanField(default=True)
    description_override = models.TextField(blank=True, null=True)

    name_override = models.CharField(max_length=100, null=True, blank=True)
    review_level: models.CharField = models.CharField(
        max_length=20, choices=ReviewLevel.choices, default=ReviewLevel.RED
    )
