from datetime import datetime
from typing import List, Optional

import graphene
from django.contrib.auth.models import User as UserModel
from django.db.models import Count, Q
from django.db.models.query import QuerySet
from django.utils import timezone
from graphene.types import generic
from graphene_django import DjangoConnectionField, DjangoObjectType
from graphql_relay import to_global_id

from core.portal.accounts.models.role import Role as RoleModel
from core.portal.accounts.permissions import USERS_READ
from core.portal.accounts.permissions_manager import (
    has_organization_permission,
    has_zone_permission,
)
from core.portal.api.models.invitation import Invitation
from core.portal.api.models.profile import Profile
from core.portal.incidents.graphql.enums import TaskStatusEnum
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.zones.models.zone import Zone


class UserStats(graphene.ObjectType):
    bookmark_total_count = graphene.Int()
    bookmark_open_count = graphene.Int()
    bookmark_resolved_count = graphene.Int()


class Role(DjangoObjectType):
    class Meta:
        model = RoleModel
        fields = [
            "id",
            "key",
            "name",
        ]

    id = graphene.ID(required=True)
    key = graphene.String(required=True)
    name = graphene.String(required=True)

    @staticmethod
    def resolve_id(parent: RoleModel, *args: None, **kwargs: None):
        del args, kwargs
        return to_global_id(Role._meta.name, parent.pk)


class InvitedUserType(
    graphene.ObjectType
):  # pylint: disable=too-few-public-methods
    """Graphene class type to represent invited user object that is returned"""

    user = graphene.Field(lambda: UserType)
    role = graphene.NonNull(Role)
    sites = graphene.List(
        graphene.lazy_import("core.portal.zones.graphql.types.ZoneType")
    )
    expired = graphene.Boolean(required=True)
    token = graphene.String(required=True)
    created_at = graphene.DateTime(required=True)


class UserType(DjangoObjectType):
    class Meta:
        model = UserModel
        interfaces = [graphene.relay.Node]
        fields = [
            "id",
            "pk",
            "first_name",
            "last_name",
            "full_name",
            "initials",
            "email",
            "groups",
        ]
        filter_fields = []

    pk = graphene.Int()
    is_active = graphene.Boolean()
    full_name = graphene.String()
    initials = graphene.String()
    roles = graphene.List(graphene.NonNull(Role))
    permissions = graphene.List(graphene.NonNull(graphene.String))
    groups = generic.GenericScalar()
    organization = graphene.Field(
        graphene.lazy_import(
            "core.portal.organizations.graphql.types.OrganizationType"
        )
    )

    zone = graphene.Field(
        graphene.lazy_import("core.portal.zones.graphql.types.ZoneType")
    )
    # TODO: deprecate `site` in favor of `zone`
    site = graphene.Field(
        graphene.lazy_import("core.portal.zones.graphql.types.ZoneType")
    )
    sites = graphene.List(
        graphene.lazy_import("core.portal.zones.graphql.types.ZoneType")
    )

    is_admin = graphene.Boolean()
    is_superuser = graphene.Boolean()
    picture = graphene.String()
    tasks_assigned_by = DjangoConnectionField(
        graphene.lazy_import("core.portal.incidents.graphql.types.TaskType"),
        status=graphene.Argument(TaskStatusEnum, required=False),
    )
    tasks_assigned_by_stats = graphene.Field(
        graphene.lazy_import(
            "core.portal.incidents.graphql.types.TaskStatsType"
        ),
        from_utc=graphene.DateTime(),
        to_utc=graphene.DateTime(),
    )
    tasks_assigned_to = DjangoConnectionField(
        graphene.lazy_import("core.portal.incidents.graphql.types.TaskType"),
        status=graphene.Argument(TaskStatusEnum, required=False),
    )
    tasks_assigned_to_stats = graphene.Field(
        graphene.lazy_import(
            "core.portal.incidents.graphql.types.TaskStatsType"
        ),
        from_utc=graphene.DateTime(),
        to_utc=graphene.DateTime(),
    )
    bookmarked_incidents = DjangoConnectionField(
        graphene.lazy_import(
            "core.portal.incidents.graphql.types.IncidentType"
        ),
    )
    stats = graphene.Field(UserStats)
    teammates = DjangoConnectionField(lambda: UserType)

    invited_users = graphene.List(graphene.NonNull(InvitedUserType))

    @staticmethod
    def resolve_teammates(
        parent: UserModel, info: graphene.ResolveInfo, **__: None
    ) -> QuerySet[UserModel]:
        user = info.context.user

        if has_organization_permission(
            user, parent.profile.current_organization, USERS_READ
        ):
            return parent.profile.current_organization.active_users.exclude(
                pk=parent.pk
            )

        if not has_zone_permission(user, user.profile.site, USERS_READ):
            raise PermissionDenied(
                "does not have permission to view users in zone"
            )

        return parent.profile.current_organization.active_users.filter(
            zones__in=Zone.objects.filter(
                users__id__exact=user.id, users__is_active=True
            ),
        ).exclude(pk=user.id)

    @staticmethod
    def resolve_invited_users(
        parent: UserModel, info: graphene.ResolveInfo, **_: None
    ) -> List[InvitedUserType]:
        """Resolves invited users.
        :param parent: the parent user model object.
        :param info: graphene resolve info object.
        :param _: extra
        :return: returns a list of invited user types.
        """
        invitations = (
            Invitation.objects.filter(
                redeemed=False,
                invitee__is_active=False,
                zones__in=Zone.objects.filter(
                    users__id__exact=info.context.user.id,
                ),
                organization=info.context.user.profile.current_organization,
            )
            .order_by("invitee", "-created_at")
            .distinct("invitee")
        )
        invited_users = []
        for invitation in invitations:
            invited_users.append(
                InvitedUserType(
                    user=invitation.invitee,
                    role=invitation.role,
                    sites=invitation.zones.all(),
                    expired=invitation.expires_at <= timezone.now(),
                    created_at=invitation.created_at,
                    token=invitation.token,
                )
            )

        return invited_users

    @staticmethod
    def resolve_is_active(parent: UserModel, info, *args):
        del info, args
        return parent.is_active

    @staticmethod
    def resolve_full_name(parent: UserModel, info, *args):
        del info, args
        return f"{parent.first_name} {parent.last_name}".strip()

    @staticmethod
    def resolve_initials(parent: UserModel, *_: None) -> str:
        initials = parent.email[0]
        if parent.first_name and parent.last_name:
            initials = f"{parent.first_name[0]}{parent.last_name[0]}"
        elif parent.first_name:
            initials = parent.first_name[0].upper()
        elif parent.last_name:
            initials = parent.last_name[0].upper()
        return initials.upper()

    @staticmethod
    def resolve_roles(parent: UserModel, info, *args) -> List[Role]:
        del info, args
        return [
            user_role.role
            for user_role in parent.user_roles.filter(
                removed_at__isnull=True
            ).all()
        ]

    @staticmethod
    def resolve_organization(parent: UserModel, info, *args):
        del info, args
        return parent.profile.current_organization

    # TODO: deprecate `site` in favor of `zone`
    @staticmethod
    def resolve_zone(parent: UserModel, info, *args):
        del info, args
        return parent.profile.site

    @staticmethod
    def resolve_site(parent: UserModel, info, *args):
        del info, args
        return parent.profile.site

    @staticmethod
    def resolve_sites(parent: UserModel, info, *args) -> List[Zone]:
        del info, args
        # TODO: support role-based access control for arbitrary sites
        if parent.is_superuser:
            # Allow superusers to switch to any site
            return parent.profile.current_organization.sites.all()
        if parent.profile.site:
            # For now, don't allow non-superusers to additional sites
            return parent.profile.current_organization.sites.filter(
                users__id=parent.id,
            ).all()
        return []

    @staticmethod
    def resolve_is_admin(parent: UserModel, info, *args):
        del info, args
        return parent.is_superuser

    @staticmethod
    def resolve_is_superuser(parent: UserModel, info, *args):
        del info, args
        return parent.is_superuser

    @staticmethod
    def resolve_picture(parent: UserModel, info, *args) -> Optional[str]:
        del info, args
        data = parent.profile.data or {}
        return data.get("avatarUrl")

    @staticmethod
    def resolve_tasks_assigned_by(
        parent: UserModel, info, *args, status: TaskStatusEnum = None, **kwargs
    ):
        del info, args, kwargs
        tasks = parent.profile.incidents_assigned_by_me.all().distinct()
        if status:
            tasks = tasks.filter(status=status.value)
        return tasks.order_by("-timestamp")

    @staticmethod
    def resolve_tasks_assigned_by_stats(
        parent: UserModel,
        info: graphene.ResolveInfo,
        from_utc: Optional[datetime] = None,
        to_utc: Optional[datetime] = None,
    ):
        del info
        incidents = parent.profile.incidents_assigned_by_me.all().distinct()
        if from_utc:
            incidents = incidents.from_timestamp(from_utc)
        if to_utc:
            incidents = incidents.to_timestamp(to_utc)
        return incidents

    @staticmethod
    def resolve_tasks_assigned_to(
        parent: UserModel, info, *args, status: TaskStatusEnum = None, **kwargs
    ):
        del info, args, kwargs
        tasks = parent.profile.incidents_assigned_to_me.all().distinct()
        if status:
            tasks = tasks.filter(status=status.value)
        return tasks.order_by("-timestamp")

    @staticmethod
    def resolve_tasks_assigned_to_stats(
        parent: UserModel,
        info: graphene.ResolveInfo,
        from_utc: Optional[datetime] = None,
        to_utc: Optional[datetime] = None,
    ):
        del info
        incidents = parent.profile.incidents_assigned_to_me.all().distinct()
        if from_utc:
            incidents = incidents.from_timestamp(from_utc)
        if to_utc:
            incidents = incidents.to_timestamp(to_utc)
        return incidents

    @staticmethod
    def resolve_bookmarked_incidents(
        parent: UserModel,
        info,
        *args,
        **kwargs,
    ):
        del info, args, kwargs
        return parent.profile.starred_list.incidents.filter(
            zone=parent.profile.site
        )

    @staticmethod
    def resolve_stats(
        parent: UserModel,
        info,
        *args,
        **kwargs,
    ):
        del info, args, kwargs
        counts = parent.profile.starred_list.incidents.filter(
            zone=parent.profile.site
        ).aggregate(
            total=Count("pk"),
            resolved=Count(
                "pk", filter=Q(status=TaskStatusEnum.RESOLVED.value)
            ),
            open=Count(
                "pk",
                filter=Q(
                    Q(status__isnull=True)
                    | Q(status=TaskStatusEnum.OPEN.value)
                ),
            ),
        )
        return UserStats(
            bookmark_total_count=counts["total"],
            bookmark_resolved_count=counts["resolved"],
            bookmark_open_count=counts["open"],
        )

    @staticmethod
    def resolve_permissions(
        parent: UserModel,
        info,
        *args,
        **kwargs,
    ) -> Optional[List[str]]:
        del args, kwargs
        if not parent.profile.permissions:
            return None

        return list(parent.profile.permissions)


# TODO: delete this in favor of placing all custom attributes on the user type
class ProfileType(DjangoObjectType):
    class Meta:
        model = Profile
        fields = [
            "id",
            "is_admin",
            "organization",
            "picture",
            "star_list_id",
        ]

    is_admin = graphene.Boolean()
    picture = graphene.String()
    star_list_id = graphene.Int()

    @staticmethod
    def resolve_picture(parent: Profile, info):
        del info
        return parent.data and parent.data.get("avatarUrl")

    @staticmethod
    def resolve_star_list_id(parent: Profile, info):
        del info
        return parent.starred_list.id


class UserError(graphene.ObjectType):
    message = graphene.String(required=True)
    code = graphene.String()
    field = graphene.List(graphene.String)
