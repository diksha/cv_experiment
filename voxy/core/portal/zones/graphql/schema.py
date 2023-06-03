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


import graphene

from core.portal.lib.graphql.utils import pk_from_global_id
from core.portal.zones.graphql.mutations import ZoneCreate
from core.portal.zones.graphql.types import ZoneType
from core.portal.zones.models.zone import Zone


class ZoneMutations(graphene.ObjectType):

    zone_create = ZoneCreate.Field()


class ZoneQueries(graphene.ObjectType):

    zone = graphene.Field(ZoneType, zone_id=graphene.String())

    def resolve_zone(
        self, info: graphene.ResolveInfo, zone_id: str
    ) -> ZoneType:
        """Returns zone given zone id
        Args:
            info (graphene.ResolveInfo): graphene.ResolveInfo
            zone_id (str): zone id
        Returns:
            ZoneType: zone model
        """
        _, zone_pk = pk_from_global_id(zone_id)
        return Zone.objects.get(pk=zone_pk)
