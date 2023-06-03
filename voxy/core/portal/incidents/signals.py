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
from django.core.cache import cache
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from core.portal.api.models.incident_type import (
    IncidentType,
    OrganizationIncidentType,
)


@receiver(
    post_save,
    sender=IncidentType,
    dispatch_uid="invalidate_incident_type_cache_on_save",
)
@receiver(
    post_delete,
    sender=IncidentType,
    dispatch_uid="invalidate_incident_type_cache_on_delete",
)
def invalidate_incident_type_cache(sender, instance, *args, **kwargs):
    if sender != IncidentType:
        return

    if instance:
        # Delete incident type entry
        cache.delete(IncidentType.cache_key(instance.id))

        # Delete all organization incident type entries
        org_incident_types = OrganizationIncidentType.objects.filter(
            incident_type=instance
        )
        cache_keys = [
            OrganizationIncidentType.cache_key(
                t.incident_type_id, t.organization_id
            )
            for t in org_incident_types
        ]
        cache.delete_many(cache_keys)


@receiver(
    post_save,
    sender=OrganizationIncidentType,
    dispatch_uid="invalidate_organization_incident_type_cache_on_save",
)
@receiver(
    post_delete,
    sender=OrganizationIncidentType,
    dispatch_uid="invalidate_organization_incident_type_cache_on_delete",
)
def invalidate_organization_incident_type_cache(
    sender, instance, *args, **kwargs
):
    if sender != OrganizationIncidentType:
        return

    if instance:
        cache.delete(
            OrganizationIncidentType.cache_key(
                instance.id, instance.organization_id
            )
        )
