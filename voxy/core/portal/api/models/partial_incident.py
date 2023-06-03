#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

from django.contrib.postgres.indexes import BTreeIndex
from django.db import models

from core.portal.api.models.incident_type import IncidentType
from core.portal.api.models.organization import Organization
from core.portal.devices.models.camera import Camera
from core.portal.lib.models.base import Model
from core.portal.zones.models.zone import Zone


class PartialIncident(Model):
    class Meta:
        db_table = "partial_incident"
        indexes = [BTreeIndex(fields=["-timestamp"])]

    uuid = models.UUIDField(unique=True, null=True, blank=True)
    title = models.CharField(max_length=100)
    timestamp = models.DateTimeField()
    # To be used the same as the data column of the Incident model
    data = models.JSONField(null=True, blank=True)
    organization = models.ForeignKey(
        Organization,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
    )
    camera = models.ForeignKey(
        Camera,
        on_delete=models.RESTRICT,
        null=True,
        blank=True,
    )
    zone = models.ForeignKey(
        Zone,
        on_delete=models.RESTRICT,
        null=True,
        blank=True,
    )
    incident_type = models.ForeignKey(
        IncidentType, on_delete=models.CASCADE, blank=True, null=True
    )
