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

import uuid

import graphene

from core.portal.api.models.organization import Organization
from core.portal.lib.graphql.mutations import BaseMutation
from core.portal.zones.enums import ZoneType as ZoneTypeEnum
from core.portal.zones.graphql.types import ZoneType
from core.portal.zones.models import Zone


class ZoneCreate(BaseMutation):
    class Arguments:
        organization_key = graphene.String(required=True)
        zone_key = graphene.String(required=True)
        zone_name = graphene.String(required=True)
        zone_type = graphene.String(required=True)

    zone = graphene.Field(ZoneType)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        organization_key: str,
        zone_key: str,
        zone_name: str,
        zone_type: str,
    ) -> "ZoneCreate":
        # Only superusers can create zones
        if not info.context.user.is_superuser:
            raise RuntimeError("User not authorized for creating zones")

        organization = Organization.objects.get(key=organization_key)
        # Organization must not have duplicate zone name / key
        if organization.zones.filter(key__iexact=zone_key).exists():
            raise RuntimeError(
                f"Zone key already exists in organization: {zone_key}"
            )
        if organization.zones.filter(name__iexact=zone_name).exists():
            raise RuntimeError(
                f"Zone name already exists in organization: {zone_name}"
            )
        if zone_type not in ZoneTypeEnum:
            raise RuntimeError(f"Zone type does not exist in db: {zone_type}")

        zone_instance = Zone.objects.create(
            name=zone_name,
            key=zone_key,
            zone_type=zone_type,
            organization=organization,
            anonymous_key=uuid.uuid4(),
        )
        return ZoneCreate(zone=zone_instance)
