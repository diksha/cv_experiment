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
from core.portal.testing.utils import random_datetime
from django.utils import timezone
from datetime import timedelta
from random import SystemRandom
from core.portal.api.models.incident import Incident, IncidentType


def generate_incidents(count: int):
    """Duplicate existing incident with slightly different attributes."""
    incident_types = list(IncidentType.objects.all())
    base_incident = Incident.objects.first()
    for _ in range(count):
        base_incident.pk = None
        base_incident.timestamp = random_datetime(timezone.now() - timedelta(days=90), timezone.now())
        base_incident.incident_type = SystemRandom().choice(incident_types)
        base_incident.priority = SystemRandom().choice(["low", "medium", "high"])
        base_incident.title = base.incident_type.name
        base_incident.save()
        print(f"saved new incident: {base.pk}")