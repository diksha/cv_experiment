import uuid

import graphene

from core.portal.api.models.organization import Organization
from core.portal.lib.graphql.mutations import BaseMutation
from core.portal.organizations.graphql.types import OrganizationType


class OrganizationCreate(BaseMutation):
    class Arguments:
        organization_key = graphene.String(required=True)
        organization_name = graphene.String(required=True)
        time_zone = graphene.String(default_value="US/Pacific")

    organization = graphene.Field(OrganizationType)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        organization_key: str,
        organization_name: str,
        time_zone: str,
    ) -> "OrganizationCreate":
        # Only superusers can create organizations
        if not info.context.user.is_superuser:
            raise RuntimeError("User not authorized for creating organization")

        # Do not create duplicate organization names / keys
        if Organization.objects.filter(key__iexact=organization_key).exists():
            raise RuntimeError("Organization key already exists")
        if Organization.objects.filter(
            name__iexact=organization_name
        ).exists():
            raise RuntimeError("Organization name already exists")

        org_instance = Organization.objects.create(
            name=organization_name,
            key=organization_key,
            timezone=time_zone,
            anonymous_key=uuid.uuid4(),
        )
        return OrganizationCreate(organization=org_instance)
