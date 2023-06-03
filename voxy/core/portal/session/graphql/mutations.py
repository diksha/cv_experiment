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

from core.portal.accounts.permissions import SELF_SWITCH_SITE
from core.portal.accounts.permissions_manager import has_zone_permission
from core.portal.api.models.incident import Incident
from core.portal.api.models.organization import Organization
from core.portal.incidents.graphql.types import IncidentType
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.lib.graphql.mutations import BaseMutation
from core.portal.lib.graphql.utils import pk_from_global_id
from core.portal.zones.models import Zone


class CurrentUserAddBookmark(BaseMutation):
    class Arguments:
        incident_id = graphene.ID(required=True)

    incident = graphene.Field(IncidentType)

    @staticmethod
    def mutate(
        root: None, info: graphene.ResolveInfo, incident_id: str
    ) -> "CurrentUserAddBookmark":
        _, pk = pk_from_global_id(incident_id)
        incident = Incident.objects.get(pk=pk)
        if (
            not incident.organization
            == info.context.user.profile.current_organization
        ):
            raise PermissionDenied(
                "Incidents may only be bookmarked by active members of the organization."
            )
        profile = info.context.user.profile
        profile.starred_list.incidents.add(incident)

        return CurrentUserAddBookmark(incident=incident)


class CurrentUserRemoveBookmark(BaseMutation):
    class Arguments:
        incident_id = graphene.ID(required=True)

    incident = graphene.Field(IncidentType)

    @staticmethod
    def mutate(
        root: None, info: graphene.ResolveInfo, incident_id: str
    ) -> "CurrentUserRemoveBookmark":
        _, pk = pk_from_global_id(incident_id)
        incident = Incident.objects.get(pk=pk)
        if (
            not incident.organization
            == info.context.user.profile.current_organization
        ):
            raise PermissionDenied(
                "Incidents may only be bookmarked by active members of the organization."
            )
        profile = info.context.user.profile
        profile.starred_list.incidents.remove(incident)

        return CurrentUserRemoveBookmark(incident=incident)


class CurrentUserOrganizationUpdate(BaseMutation):
    class Arguments:
        organization_id = graphene.ID(required=True)

    status = graphene.Boolean()

    @staticmethod
    def mutate(root, info, organization_id):
        if not info.context.user.is_superuser:
            raise RuntimeError("You are not allowed to change organization")

        profile = info.context.user.profile
        _, pk = pk_from_global_id(organization_id)
        organization = Organization.objects.get(pk=pk)
        profile.organization = organization
        profile.site = organization.sites.first()
        profile.save()
        return CurrentUserOrganizationUpdate(status=True)


class CurrentUserSiteUpdate(BaseMutation):
    class Arguments:
        site_id = graphene.ID(required=True)

    status = graphene.Boolean()
    new_site_id = graphene.ID()

    @staticmethod
    def mutate(root, info, site_id):
        del root
        _, pk = pk_from_global_id(site_id)

        if not has_zone_permission(
            info.context.user, Zone.objects.get(pk=pk), SELF_SWITCH_SITE
        ):
            return PermissionDenied("You are not allowed to change zones.")

        profile = info.context.user.profile
        zone = Zone.objects.get(pk=pk)
        profile.site = zone
        profile.organization = zone.organization
        profile.save()
        return CurrentUserSiteUpdate(status=True, new_site_id=zone.id)
