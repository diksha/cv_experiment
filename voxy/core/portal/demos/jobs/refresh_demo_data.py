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
from datetime import timedelta

from django.utils import timezone
from loguru import logger

from core.portal.api.models.organization import Organization
from core.portal.lib.jobs.base import JobBase
from core.portal.lib.utils.date_utils import hours_between

DEMO_ORG_KEY = "VOXEL_DEMO"


class RefreshDemoDataJob(JobBase):
    """Refreshes demo organization data.

    This job is intended to run hourly and bump demo data timestamps by 1 hour.

    It is important that incident timestamps stay in the same relative order
    as sales demo scripts expect the data to be presented in a specific order.
    """

    def run(self) -> None:
        demo_org = Organization.objects.filter(
            key=DEMO_ORG_KEY, is_sandbox=True
        ).first()

        if not demo_org:
            raise RuntimeError("No demo organization found")

        # Don't bump timestamps on RIMS incidents
        incidents_queryset = demo_org.incidents.exclude(
            camera__uuid="rims/atlanta/0001/cha"
        )

        most_recent_incident = incidents_queryset.order_by("timestamp").last()

        if not most_recent_incident:
            raise RuntimeError("No incidents found in demo organization")

        now = timezone.now()

        if hours_between(most_recent_incident.timestamp, now) >= 1:
            logger.info("Refreshing demo data...")
            for incident in incidents_queryset:
                incident.timestamp = incident.timestamp + timedelta(hours=1)
                incident.save()
                # Update timestamps for all the incident's comments
                for comment in incident.comments.all():
                    updated_created_at = comment.created_at + timedelta(
                        hours=1
                    )

                    # Don't allow comments timestamps to be in the future
                    comment.created_at = min(now, updated_created_at)
                    comment.save()
            logger.info("Finished refreshing demo data")
        else:
            logger.info(
                "Skipping demo data refresh, data was updated within the last hour"
            )
