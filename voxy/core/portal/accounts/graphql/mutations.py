from typing import List, Optional

import graphene
from auth0.v3.exceptions import Auth0Error
from django.contrib.auth.models import Group, User
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import transaction
from django.shortcuts import get_object_or_404
from django.utils import timezone
from loguru import logger

from core.portal.accounts.clients.auth0 import Auth0ManagementClient
from core.portal.accounts.graphql.types import UserType
from core.portal.accounts.models.role import Role
from core.portal.accounts.models.user_role import UserRole
from core.portal.accounts.permissions import (
    REVIEWER_ACCOUNT_CREATE,
    REVIEWER_ACCOUNTS_UPDATE_ROLE,
    SELF_UPDATE_PROFILE,
    USERS_INVITE,
    USERS_REMOVE,
    USERS_UPDATE_ROLE,
    USERS_UPDATE_SITE,
)
from core.portal.accounts.permissions_manager import (
    has_global_permission,
    has_zone_permission,
)
from core.portal.accounts.roles import INTERNAL_REVIEWER
from core.portal.accounts.services import create_auth0_user
from core.portal.api.models.invitation import Invitation
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.lib.graphql.mutations import BaseMutation
from core.portal.lib.graphql.utils import pk_from_global_id
from core.portal.zones.models import Zone


class UserRemove(BaseMutation):
    class Arguments:
        user_id = graphene.ID(required=True)

    status = graphene.Boolean()

    @staticmethod
    def mutate(root, info, user_id):
        del root

        _, pk = pk_from_global_id(user_id)

        # The current user should have access to all of the same
        # zones as the user being removed. This should generally
        # be true, but in edge cases where the affected user has
        # access outside of the current user's visibility/access,
        # we don't want to allow the current user to affect it.
        # Instead, they can remove the affected user from all
        # sites where they have overlapping access.
        affected_user = User.objects.filter(pk=pk, is_active=True).first()
        if not affected_user or not has_zone_permission(
            info.context.user,
            affected_user.zones.all(),
            USERS_REMOVE,
        ):
            raise PermissionDenied(
                "You do not have permission to remove this user."
            )

        affected_user.is_active = False
        affected_user.save()

        return UserRemove(status=True)


class UserRoleUpdate(BaseMutation):
    class Arguments:
        user_id = graphene.ID(required=True)
        role_id = graphene.ID(required=True)

    status = graphene.Boolean()

    @staticmethod
    def conduct_permission_check(
        info: graphene.ResolveInfo,
        current_user_role: UserRole,
        requested_role: Role,
        requested_user_id: str,
    ):
        zones = Zone.objects.filter(
            users__id__in=[info.context.user.id, requested_user_id],
            users__is_active=True,
        )
        if len(list(zones)) == 0:
            return ValueError(
                "no zone membership overlap.",
            )

        if not has_zone_permission(
            info.context.user, zones[0], USERS_UPDATE_ROLE
        ):
            return PermissionDenied(
                "You do not have permission to modify roles."
            )

        return None

    @staticmethod
    def mutate(root, info, user_id, role_id):
        del root

        _, user_pk = pk_from_global_id(user_id)
        _, role_pk = pk_from_global_id(role_id)

        requested_role = Role.objects.get(pk=role_pk)
        current_user_role = UserRole.objects.get(
            user__pk=user_pk, removed_at__isnull=True, user__is_active=True
        )

        permission_error = UserRoleUpdate.conduct_permission_check(
            info=info,
            current_user_role=current_user_role,
            requested_role=requested_role,
            requested_user_id=user_pk,
        )
        if permission_error:
            raise permission_error

        if requested_role == current_user_role:
            return UserRoleUpdate(status=True)

        now = timezone.now()

        current_user_role.removed_at = now
        current_user_role.removed_by = info.context.user
        current_user_role.save()

        UserRole.objects.create(
            user=User.objects.get(pk=user_pk, is_active=True),
            role=requested_role,
            assigned_by=info.context.user,
            assigned_at=now,
        )

        return UserRoleUpdate(
            status=True,
        )


class UserZonesUpdate(BaseMutation):
    class Arguments:
        user_id = graphene.ID(required=True)
        zones = graphene.List(graphene.ID, required=True)

    status = graphene.Boolean()

    @staticmethod
    def mutate(root, info, user_id, zones):
        del root

        zone_pks = []
        for zone_id in zones:
            _, zone_pk = pk_from_global_id(zone_id)
            zone_pks.append(zone_pk)

        _, user_pk = pk_from_global_id(user_id)
        user_zones = set(
            Zone.objects.filter(
                users__id__exact=user_pk,
                users__is_active=True,
            )
        )

        requested_zones = set(Zone.objects.filter(id__in=zone_pks))

        # left and right side of ven diagram
        zones_to_remove = list(user_zones.difference(requested_zones))
        zones_to_add = list(requested_zones.difference(user_zones))

        if not has_zone_permission(
            info.context.user,
            zones_to_remove,
            USERS_UPDATE_SITE,
        ):
            raise PermissionDenied(
                "You do not have permission to update the user of this zone."
            )

        user = User.objects.get(pk=user_pk)
        user_profile_removed = False
        for zone in zones_to_remove:
            if zone == user.profile.site:
                user_profile_removed = True

            zone.users.remove(user)
            zone.save()

        if not has_zone_permission(
            info.context.user,
            zones_to_add,
            USERS_UPDATE_SITE,
        ):
            raise PermissionDenied(
                "You do not have permission to update the user of this zone."
            )

        for zone in zones_to_add:
            zone.users.add(user)
            zone.save()

        if user_profile_removed:
            zones = Zone.objects.filter(
                users__id__exact=user_pk,
                users__is_active=True,
                organization=user.profile.current_organization,
            )
            user.profile.site = zones.first()

        return UserZonesUpdate(
            status=True,
        )


class UserUpdate(BaseMutation):
    class Arguments:
        user_id = graphene.ID(required=True)
        roles = graphene.List(graphene.String)
        first_name = graphene.String()
        last_name = graphene.String()
        is_active = graphene.Boolean()

    user = graphene.Field(UserType)

    @classmethod
    def mutate(
        cls,
        root,
        info,
        *args,
        user_id,
        roles=None,
        first_name=None,
        last_name=None,
        is_active=None,
        **kwargs,
    ):
        del root, args, kwargs
        _, pk = pk_from_global_id(user_id)
        user = get_object_or_404(User, pk=pk)
        # TODO: implement permission-based authorization mechanism
        is_self = user == info.context.user
        is_admin = info.context.user.groups.filter(name="admin").exists()
        is_same_org = (
            user.profile.current_organization
            != info.context.user.profile.current_organization
        )

        if not is_self:
            if not is_admin and is_same_org:
                raise PermissionDenied(
                    "You are not allowed to update this user."
                )

        if first_name:
            user.first_name = first_name
        if last_name:
            user.last_name = last_name
        if roles is not None:
            groups = Group.objects.filter(
                name__in=[role.lower() for role in roles]
            )
            user.groups.set(groups)
        if is_active is not None:
            user.is_active = is_active

        user.save()

        return UserUpdate(user=user)


class UserResendInvitation(BaseMutation):
    """Ability for a user to resend an invitation to a user to register."""

    class Arguments:
        """Arguments is a subclass that contains the args for this mutation."""

        invitation_token = graphene.String(required=True)

    status = graphene.Boolean()

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        invitation_token: str,
    ) -> "UserResendInvitation":
        """Mutation to resend an invitation.

        Args:
            root (None): root obj
            info (graphene.ResolveInfo): info object from graphene
            invitation_token (str): the token of the resubmitted invitation

        Returns:
            UserResendInvitation: mutation response with status code

        Raises:
            PermissionDenied: if failed permission checks
        """

        del root

        existing_invitation = Invitation.objects.get(
            token=invitation_token,
        )
        zones = existing_invitation.zones.all()

        if not has_zone_permission(
            info.context.user,
            zones,
            USERS_INVITE,
        ):
            raise PermissionDenied(
                "You do not have permission to invite users into this zone."
            )

        existing_invitation.resend(invited_by=info.context.user)

        return UserResendInvitation(
            status=True,
        )


class InvitationInputSchema(graphene.InputObjectType):
    email = graphene.String(required=True)
    role_id = graphene.String(required=True)
    zone_ids = graphene.List(graphene.String, required=True)


class UserInvite(BaseMutation):
    """Ability for a user to invite a user to register."""

    class Arguments:
        invitees = graphene.List(InvitationInputSchema, required=True)

    status = graphene.Boolean()

    @staticmethod
    def process_user_invitation(
        info: graphene.ResolveInfo,
        requesting_user: User,
        invitee: InvitationInputSchema,
    ):
        """Helper function to process user invitation one at a time.

        Args:
            info (graphene.ResolveInfo): info object from graphene
            requesting_user (User): the user making the network request
            invitee (InvitationInputSchema): the person that they are requesting to invite

        Raises:
            PermissionDenied: if failed permission checks
            ValueError: when email does not match valid regex expression
        """

        _, role_pk = pk_from_global_id(invitee["role_id"])

        zone_pks = []
        for zone_id in invitee["zone_ids"]:
            _, zone_pk = pk_from_global_id(zone_id)
            zone_pks.append(zone_pk)

        zones = list(Zone.objects.filter(id__in=zone_pks))

        if not has_zone_permission(
            requesting_user,
            zones,
            USERS_INVITE,
        ):
            raise PermissionDenied(
                "You do not have permission to invite users into this zone."
            )

        email = invitee["email"]

        try:
            validate_email(email)
        except ValidationError as error:
            raise ValueError(
                f"{email} is not a valid email address"
            ) from error

        Invitation.send(
            invited_by=info.context.user,
            email=email,
            role=Role.objects.get(pk=role_pk),
            organization=info.context.user.profile.current_organization,
            zones=zones,
        )

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        invitees: List[InvitationInputSchema],
    ) -> "UserInvite":
        """Mutation to process bulk user invitations.
        :param root: root obj
        :param info: info object from graphene
        :param invitees: the array of invitees to send invites to
        :return: returns status code true if processing all invites
        """
        del root
        for invitee in invitees:
            UserInvite.process_user_invitation(
                info=info,
                requesting_user=info.context.user,
                invitee=invitee,
            )

        return UserInvite(status=True)


class ReviewerAccountCreate(BaseMutation):
    """Reviewer account creation mutation."""

    class Arguments:
        """Mutation arguments."""

        email = graphene.String(required=True)
        name = graphene.String(required=True)
        password = graphene.String(required=True)

    user = graphene.Field(UserType)

    @classmethod
    def mutate(
        cls,
        root: None,
        info: graphene.ResolveInfo,
        email: str,
        name: str,
        password: str,
    ) -> "ReviewerAccountCreate":
        """Create a new reviewer account

        Args:
            root (None): graphene root object
            info (graphene.ResolveInfo): graphene context
            email (str): email address for new account
            name (str): name of reviewer
            password (str): account password

        Raises:
            RuntimeError: when argument validation fails
            RuntimeError: when account creation fails
            PermissionDenied: when requesting user does not have sufficient permissions

        Returns:
            ReviewerAccountCreate: new user instance
        """
        del root

        if not has_global_permission(
            info.context.user, REVIEWER_ACCOUNT_CREATE
        ):
            raise PermissionDenied(
                "You do not have permission to create reviewer accounts."
            )

        if not email:
            raise RuntimeError("email is required")
        if not name:
            raise RuntimeError("name is required")
        if not password:
            raise RuntimeError("password is required")
        if User.objects.filter(email=email).count() > 0:
            raise RuntimeError(f"User already exists for email: {email}")

        email = email.lower()
        last_name = "(Reviewer)"

        with transaction.atomic():
            auth0_id = create_auth0_user(
                email,
                name,
                last_name,
                password,
            )
            user = User.objects.create(
                username=email,
                email=email,
                first_name=name,
                last_name=last_name,
            )
            user.profile.data = {"auth0_id": auth0_id}
            user.profile.save()
            UserRole.objects.create(
                user=user,
                role=Role.objects.get(key=INTERNAL_REVIEWER),
            )
            return ReviewerAccountCreate(user=user)


class UserNameUpdate(BaseMutation):
    class Arguments:
        """Arguments needed for UserNameUpdate"""

        user_id = graphene.String(required=True)
        first_name = graphene.String()
        last_name = graphene.String()

    user = graphene.Field(UserType)

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        user_id: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> "UserNameUpdate":
        """Updates user name in both database and auth0

        Args:
            root (None): root object
            info (graphene.ResolveInfo): info context object of req
            user_id (str): user to be updated
            first_name (Optional[str], optional): First name. Defaults to None.
            last_name (Optional[str], optional): Last name. Defaults to None.

        Raises:
            PermissionDenied: If does not have permission to change
            auth0_error: If network/configuration errors with auth0
            RuntimeError: If response warped from auth0
            RuntimeError: If request user id is for a google account

        Returns:
            UserNameUpdate: instance of mutation
        """
        del root
        user = User.objects.get(profile__data__auth0_id=user_id)
        is_self = user == info.context.user

        if "google-oauth-2" in user_id:
            raise RuntimeError("Cannot update Google-associated names.")

        if (
            not has_global_permission(info.context.user, SELF_UPDATE_PROFILE)
            or not is_self
        ):
            raise PermissionDenied("You are not allowed to update this user.")

        with transaction.atomic():
            auth0_client = Auth0ManagementClient()

            update_user_body = {}
            if first_name:
                update_user_body["given_name"] = first_name
                user.first_name = first_name

            if last_name:
                update_user_body["family_name"] = last_name
                user.last_name = last_name

            if first_name and last_name:
                update_user_body["name"] = f"{first_name} {last_name}".strip()

            try:
                update_response = auth0_client.users.update(
                    user_id, update_user_body
                )
            except Auth0Error as auth0_error:
                logger.exception("Failed to update Auth0 user")
                raise auth0_error

            if not update_response or not update_response.get("user_id"):
                logger.error(
                    f"Failed to update user in Auth0: {update_response}"
                )
                raise RuntimeError("Something went wrong while updating")

            user.save()

        return UserNameUpdate(user=user)


class UserMFAUpdate(BaseMutation):
    class Arguments:
        """Arguments needed for UserMFAUpdate"""

        user_id = graphene.String(required=True)
        toggled_mfa_on = graphene.Boolean(required=True)

    success = graphene.Boolean()

    @staticmethod
    def mutate(
        root: None,
        info: graphene.ResolveInfo,
        user_id: str,
        toggled_mfa_on: bool = False,
    ) -> "UserMFAUpdate":
        """Updates MFA status on user's account

        Args:
            root (None): root object
            info (graphene.ResolveInfo): info context object of req
            user_id (int): user to be updated
            toggled_mfa_on (bool, optional): whether mfa being toggled on or not. Defaults to False.

        Raises:
            PermissionDenied: If does not have permission to change
            auth0_error: If network/configuration errors with auth0
            RuntimeError: If response warped from auth0

        Returns:
            UserMFAUpdate: instance of mutation
        """
        del root

        user = User.objects.get(profile__data__auth0_id=user_id)
        is_self = user == info.context.user

        if (
            not has_global_permission(info.context.user, SELF_UPDATE_PROFILE)
            or not is_self
        ):
            raise PermissionDenied("You are not allowed to update this user.")

        auth0_client = Auth0ManagementClient()

        try:
            update_response = auth0_client.users.update(
                user_id,
                {
                    "user_metadata": {
                        "has_mfa": toggled_mfa_on,
                    }
                },
            )
        except Auth0Error as auth0_error:
            logger.exception("Failed to update Auth0 user")
            raise auth0_error

        if not update_response or not update_response.get("user_id"):
            logger.error(f"Failed to update user in Auth0: {update_response}")
            raise RuntimeError("Something went wrong while updating")

        if not toggled_mfa_on:
            # Delete authenticators
            try:
                auth0_client.users.delete_authenticators(user_id)
            except Auth0Error as auth0_error:
                logger.exception("Failed to delete authenticators")
                raise auth0_error

        return UserMFAUpdate(success=True)


# right now just a toggle on shadow review
class ReviewerAccountRoleUpdate(BaseMutation):
    class Arguments:
        user_id = graphene.ID(required=True)

    status = graphene.Boolean()

    @staticmethod
    def mutate(
        root: None, info: graphene.ResolveInfo, user_id: str
    ) -> "ReviewerAccountRoleUpdate":
        """Update reviewer user role to either trainee or operator

        Args:
            root (None): root object
            info (graphene.ResolveInfo): info context object of req
            user_id (str): user to be updated

        Returns:
            ReviewerAccountRoleUpdate: reviewer user role update
        """
        del root

        _, user_pk = pk_from_global_id(user_id)

        if not has_global_permission(
            info.context.user, REVIEWER_ACCOUNTS_UPDATE_ROLE
        ):
            return PermissionDenied(
                "You do not have permission to modify roles."
            )

        user = User.objects.get(pk=user_pk)

        current_value = user.profile.data.get("shadow_reviewer", False)
        user.profile.data["shadow_reviewer"] = not current_value
        user.profile.save()

        return ReviewerAccountRoleUpdate(
            status=True,
        )
