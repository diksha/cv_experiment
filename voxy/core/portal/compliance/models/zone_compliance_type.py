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
from django.contrib import admin
from django.db import models

from core.portal.compliance.models.compliance_type import ComplianceType
from core.portal.lib.models.base import Model
from core.portal.zones.models.zone import Zone


class ZoneComplianceType(Model):
    """Compliance type for a specific zone."""

    class Meta:
        """Django meta class used to configure the model."""

        app_label = "compliance"
        db_table = "zone_compliance_type"
        unique_together = (
            "compliance_type",
            "zone",
        )

    enabled = models.BooleanField(default=True, null=False)
    name_override = models.CharField(
        help_text="Name field which overrides the compliance type name for a particular zone.",
        max_length=250,
        null=True,
        blank=True,
    )
    compliance_type = models.ForeignKey(
        ComplianceType,
        null=False,
        on_delete=models.CASCADE,
        related_name="zone_compliance_types",
    )
    zone = models.ForeignKey(
        Zone,
        null=False,
        on_delete=models.CASCADE,
        related_name="zone_compliance_types",
    )

    def __str__(self) -> str:
        """String representation of a model instance.

        Returns:
            str: string representation
        """
        name = self.name_override or self.compliance_type.name
        return f"{self.zone.name} - {name} ({self.compliance_type.key})"

    @property
    def name(self) -> str:
        """Zone-specific compliance type name.

        Returns:
            str: zone-specific compliance type name
        """
        return self.name_override or self.compliance_type.name


admin.site.register(ZoneComplianceType)
