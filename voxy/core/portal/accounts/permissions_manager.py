from typing import List, Union

from django.contrib.auth.models import User

from core.portal.accounts.permissions import Permission
from core.portal.api.models.organization import Organization
from core.portal.zones.models.zone import Zone


def has_zone_permission(
    user: User, zones: Union[Zone, List[Zone]], permission: Permission
) -> bool:
    if isinstance(zones, Zone):
        zones = [zones]

    zone_ids = []
    organizations = []

    for z in zones:
        organizations.append(z.organization)
        zone_ids.append(z.id)

    if has_organization_permission(user, organizations, permission):
        return True

    user_has_zone_memberships = set(
        Zone.objects.filter(
            id__in=zone_ids,
            users__id__exact=user.id,
            users__is_active=True,
        ).values_list("id", flat=True)
    ) == set(zone_ids)

    return (
        permission.zone_scope
        and permission.zone_permission_key in user.permissions
        and user_has_zone_memberships
    )


def has_organization_permission(
    user: User,
    organizations: Union[Organization, List[Organization]],
    permission: Permission,
) -> bool:
    if has_global_permission(user, permission):
        return True

    if isinstance(organizations, Organization):
        organizations = [organizations]

    organization_ids = [o.id for o in organizations]

    user_has_organization_memberships = set(
        Organization.objects.filter(
            id__in=organization_ids,
            users__id__exact=user.id,
            users__is_active=True,
        ).values_list(
            "id",
            flat=True,
        )
    ) == set(organization_ids)

    return (
        permission.organization_scope
        and permission.organization_permission_key in user.permissions
        and user_has_organization_memberships
    )


def has_global_permission(user: User, permission: Permission) -> bool:
    return (
        permission.global_scope
        and permission.global_permission_key in user.permissions
    )
