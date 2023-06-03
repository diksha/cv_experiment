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

from core.portal.compliance.enums import ComplianceTypeKey
from core.portal.lib.models.base import Model


class ComplianceType(Model):
    """Compliance type."""

    class Meta:
        """Django meta class used to configure the model."""

        app_label = "compliance"
        db_table = "compliance_type"

    key = models.CharField(
        max_length=100,
        unique=True,
        null=False,
        choices=ComplianceTypeKey.choices,
    )
    name = models.CharField(
        help_text="User friendly name displayed throughout apps.",
        max_length=250,
        null=False,
        blank=False,
    )

    def __str__(self) -> str:
        """String representation of a model instance.

        Returns:
            str: string representation
        """
        return self.key


admin.site.register(ComplianceType)
