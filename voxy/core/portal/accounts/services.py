from typing import Optional

from auth0.v3.exceptions import Auth0Error
from django.contrib.auth.models import User
from django.db import DatabaseError, transaction
from django.utils import timezone
from loguru import logger

from core.portal.accounts.clients.auth0 import Auth0ManagementClient
from core.portal.accounts.models.user_role import UserRole
from core.portal.api.models.invitation import Invitation
from core.portal.api.models.list import STARRED_LIST_NAME, List
from core.portal.api.models.profile import Profile


def initialize_account(user_id: int) -> None:
    Profile.objects.create(owner_id=user_id)
    initialize_starred_list(user_id)


def initialize_starred_list(user_id: int) -> None:
    List.objects.create(
        owner_id=user_id, name=STARRED_LIST_NAME, is_starred_list=True
    )


def register_user(
    auth0_client: Auth0ManagementClient,
    invitation: Invitation,
    first_name: str,
    last_name: str,
    password: str,
) -> User:
    """Registers a user given a valid invitation.

    Args:
        auth0_client (Auth0ManagementClient): Auth0 management client
        invitation (Invitation): invitation to redeem
        first_name (str): user's first name
        last_name (str): user's last name
        password (str): user's desired password

    Raises:
        RuntimeError: when the invitation is redeemed or expired
        RuntimeError: when the Auth0 API call fails

    Returns:
        User: the registered user instance
    """

    if (
        invitation.redeemed
        or invitation.expires_at < timezone.now()
        or invitation.invitee.is_active
    ):
        raise RuntimeError("This invitation is no longer valid.")

    new_user = invitation.invitee
    auth0_id = None

    try:
        with transaction.atomic():
            auth0_id = create_auth0_user(
                new_user.email,
                first_name,
                last_name,
                password,
                auth0_client=auth0_client,
            )
            zones = invitation.zones.all()
            role = invitation.role
            organization = invitation.organization

            UserRole.objects.create(
                user=new_user,
                role=role,
                assigned_by=invitation.invited_by,
                assigned_at=timezone.now(),
            )

            organization.users.add(new_user)
            new_user.zones.add(*zones)
            new_user.first_name = first_name
            new_user.last_name = last_name
            new_user.is_active = True
            new_user.save()

            new_user.profile.organization = organization
            new_user.profile.site = zones[0] if zones else None
            new_user.profile.data = {
                "auth0_id": auth0_id,
                "receive_daily_summary_emails": True,
            }
            new_user.profile.save()

            # Update invitation
            invitation.expires_at = timezone.now()
            invitation.redeemed = True
            invitation.save()

            return new_user

    except Auth0Error as auth0_error:
        logger.exception("Failed to create Auth0 user")
        raise auth0_error
    except (DatabaseError, RuntimeError) as error:
        # Clean up Auth0 user if transaction fails
        if auth0_id:
            clean_up_auth0_user(auth0_client, auth0_id)
        raise error


def create_auth0_user(
    email: str,
    first_name: str,
    last_name: str,
    password: str,
    auth0_client: Optional[Auth0ManagementClient] = None,
) -> str:
    """Creates a user in Auth0.

    Args:
        email (str): user's email address
        first_name (str): user's first name
        last_name (str): user's last name
        password (str): user's desired password
        auth0_client (Optional[Auth0ManagementClient]): Auth0 management client

    Raises:
        RuntimeError: when the Auth0 API returns an unexpected response
        Auth0Error: when the Auth0 API returns an error

    Returns:
        str: the newly created user's Auth0 ID
    """
    if not auth0_client:
        auth0_client = Auth0ManagementClient()

    create_response = auth0_client.users.create(
        {
            "connection": "Username-Password-Authentication",
            "email": email,
            "given_name": first_name,
            "family_name": last_name,
            "name": f"{first_name} {last_name}".strip(),
            "password": password,
            # User should receive invitation via email so consider it valid
            "email_verified": True,
        }
    )

    # TODO(troy): add more granular error handling
    if not create_response or not create_response.get("user_id"):
        logger.error(f"Failed to create user in Auth0: {create_response}")
        raise RuntimeError("Something went wrong while registering")

    auth0_id = create_response.get("user_id")
    return auth0_id


def clean_up_auth0_user(
    auth0_client: Auth0ManagementClient, auth0_id: str
) -> None:
    """Cleans up an Auth0 user after a failed transaction.

    Args:
        auth0_client (Auth0ManagementClient): Auth0 management client
        auth0_id (str): Auth0 user ID
    """
    try:
        active_user_ids = User.objects.filter(
            profile__data__auth0_id=auth0_id
        ).values_list("id", flat=True)
        if active_user_ids:
            logger.log(
                "SKipping Auth0 user cleanup because the following Django"
                f" user IDs are linked to this Auth0 Account: {active_user_ids}"
            )
        else:
            auth0_client.users.delete(auth0_id)
    except Exception:  # trunk-ignore(pylint/W0703): allow catching broad exception
        logger.exception(
            "Failed to clean up Auth0 user after transaction failure"
        )
