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
from typing import Optional

import graphene
from graphene_django import DjangoConnectionField

from core.portal.api.models.organization import (
    Organization as OrganizationModel,
)
from core.portal.lib.graphql.utils import pk_from_global_id
from core.portal.organizations.graphql.mutations import OrganizationCreate
from core.portal.organizations.graphql.types import OrganizationType


class OrganizationQueries(graphene.ObjectType):

    organizations = DjangoConnectionField(OrganizationType)
    organization = graphene.Field(
        OrganizationType,
        organization_id=graphene.ID(),
        organization_key=graphene.String(),
    )

    def resolve_organization(
        self,
        info: graphene.ResolveInfo,
        organization_id: graphene.ID,
        organization_key: graphene.String,
    ) -> Optional[OrganizationModel]:
        # Allow fetching by organization key and/or graphql global ID
        if not organization_id and not organization_key:
            raise ValueError(
                "One of organization_id or organization_key are required."
            )

        queryset = OrganizationModel.objects.all()
        if organization_id:
            _, organization_pk = pk_from_global_id(organization_id)
            queryset = queryset.filter(id=organization_pk)
        if organization_key:
            queryset = queryset.filter(key=organization_key)
        org = queryset.get()

        if info.context.user.is_superuser:
            return org
        if org and org.active_users.filter(id=info.context.user.id).exists():
            return org
        return None


class OrganizationMutations(graphene.ObjectType):

    organization_create = OrganizationCreate.Field()
