import mock
import pytest
from django.contrib.auth.models import User
from django.utils import timezone
from graphql_relay import to_global_id

from core.portal.accounts.graphql.mutations import (
    ReviewerAccountRoleUpdate,
    UserInvite,
    UserMFAUpdate,
    UserNameUpdate,
    UserRemove,
    UserResendInvitation,
    UserRoleUpdate,
    UserZonesUpdate,
)
from core.portal.accounts.models.user_role import UserRole
from core.portal.accounts.permissions import (
    REVIEWER_ACCOUNTS_UPDATE_ROLE,
    SELF_UPDATE_PROFILE,
    USERS_INVITE,
    USERS_REMOVE,
    USERS_UPDATE_ROLE,
    USERS_UPDATE_SITE,
)
from core.portal.accounts.roles import EXTERNAL_ADMIN
from core.portal.api.models.invitation import Invitation
from core.portal.testing.factories import (
    InvitationFactory,
    OrganizationFactory,
    RoleFactory,
    UserFactory,
    UserRoleFactory,
    ZoneFactory,
)
from core.portal.zones.models import Zone


@pytest.mark.django_db
@mock.patch("core.portal.accounts.graphql.mutations.pk_from_global_id")
def test_account_remove_user(mock_pk_from_global_id) -> None:
    user = UserFactory()
    org = OrganizationFactory(
        users=[user],
    )

    mock_pk_from_global_id.return_value = (None, user.pk)

    res = UserRemove.mutate(
        root=None,
        info=mock.Mock(
            context=mock.Mock(
                user=mock.Mock(
                    id=user.pk,
                    permissions=[USERS_REMOVE.organization_permission_key],
                    profile=mock.Mock(
                        current_organization=org,
                    ),
                ),
            )
        ),
        user_id=user.pk,
    )

    updated_user = User.objects.get(pk=user.pk)

    assert isinstance(res, UserRemove)
    assert updated_user.is_active is False


@pytest.mark.django_db
def test_account_update_user_role() -> None:
    requesting_user = UserFactory()
    user_to_change = UserFactory()

    zone = ZoneFactory()
    zone.users.add(requesting_user)
    zone.users.add(user_to_change)

    requesting_user.permissions = [USERS_UPDATE_ROLE.global_permission_key]

    role = RoleFactory()
    UserRoleFactory(
        user=user_to_change,
        role=RoleFactory(
            key=EXTERNAL_ADMIN,
        ),
    )

    old_user_role = UserRole.objects.filter(
        user=user_to_change, removed_at__isnull=True
    )
    assert len(old_user_role) == 1

    res = UserRoleUpdate.mutate(
        root=None,
        info=mock.Mock(
            context=mock.Mock(
                user=requesting_user,
            )
        ),
        user_id=to_global_id("User", user_to_change.pk),
        role_id=to_global_id("Role", role.pk),
    )

    updated_user_role = UserRole.objects.filter(
        user=user_to_change, removed_at__isnull=True
    )

    assert len(updated_user_role) == 1
    assert updated_user_role[0].role.key != old_user_role[0].role.key
    assert isinstance(res, UserRoleUpdate)


@pytest.mark.django_db
@mock.patch("core.portal.accounts.graphql.mutations.pk_from_global_id")
def test_account_update_user_site(mock_pk_from_global_id) -> None:
    requesting_user = UserFactory()
    requesting_user.permissions = [USERS_UPDATE_SITE.global_permission_key]

    user_to_change = UserFactory()
    zone_to_remove = ZoneFactory()
    zone_to_add = ZoneFactory()

    zone_to_remove.users.add(user_to_change)

    added_zones_users = list(
        Zone.objects.filter(pk=zone_to_add.pk, users__id=user_to_change.pk)
    )
    removed_zones_users = list(
        Zone.objects.filter(pk=zone_to_remove.pk, users__id=user_to_change.pk)
    )
    assert len(added_zones_users) == 0
    assert len(removed_zones_users) == 1

    mock_pk_from_global_id.side_effect = [
        (None, zone_to_add.pk),
        (None, user_to_change.pk),
    ]

    res = UserZonesUpdate.mutate(
        root=None,
        info=mock.Mock(
            context=mock.Mock(
                user=requesting_user,
            )
        ),
        user_id=user_to_change.pk,
        zones=[zone_to_add.pk],
    )

    added_zones_users = list(
        Zone.objects.filter(pk=zone_to_add.pk, users__id=user_to_change.pk)
    )
    removed_zones_users = list(
        Zone.objects.filter(pk=zone_to_remove.pk, users__id=user_to_change.pk)
    )
    assert isinstance(res, UserZonesUpdate)
    assert len(added_zones_users) == 1
    assert len(removed_zones_users) == 0


@pytest.mark.django_db
@mock.patch("core.portal.accounts.graphql.mutations.pk_from_global_id")
def test_account_invite_user(mock_pk_from_global_id: mock.Mock) -> None:
    """Test to verify we can invite users.
    :param mock_pk_from_global_id: mock pk getter fn
    """
    mock_email = "test_email@example.com"

    requesting_user = UserFactory()
    org = OrganizationFactory(users=[requesting_user])

    role = RoleFactory()

    zone = ZoneFactory()
    zone.users.add(requesting_user)

    mock_pk_from_global_id.side_effect = [
        (None, role.pk),
        (None, zone.pk),
        (None, role.pk),
        (None, zone.pk),
    ]

    requesting_user.permissions = [USERS_INVITE.zone_permission_key]
    requesting_user.profile.organization = org

    UserInvite.process_user_invitation(
        info=mock.Mock(
            context=mock.Mock(
                user=requesting_user,
            )
        ),
        requesting_user=requesting_user,
        invitee={
            "role_id": role.pk,
            "zone_ids": [zone.pk],
            "email": mock_email,
        },
    )

    users = list(User.objects.filter(email=mock_email))

    assert len(users) == 1
    assert users[0].is_active is False

    invitation = list(Invitation.objects.filter(invitee__email=mock_email))

    assert len(invitation) == 1
    assert invitation[0].invitee == users[0]

    # should only be able to invite user once
    try:
        UserInvite.process_user_invitation(
            info=mock.Mock(
                context=mock.Mock(
                    user=requesting_user,
                )
            ),
            requesting_user=requesting_user,
            invitee={
                "role_id": role.pk,
                "zone_ids": [zone.pk],
                "email": mock_email,
            },
        )
    except RuntimeError as runtime_error:
        assert (
            str(runtime_error)
            == "Cannot reinvite a user who has a valid invitation."
        )

    invitation = list(Invitation.objects.filter(invitee__email=mock_email))

    assert len(invitation) == 1
    assert invitation[0].invitee == users[0]


@pytest.mark.django_db
def test_account_reinvite_user() -> None:
    """Test to verify we can re-invite a particular user."""
    requesting_user = UserFactory()

    zone = ZoneFactory()
    zone.users.add(requesting_user)

    requesting_user.permissions = [USERS_INVITE.zone_permission_key]

    invitation = InvitationFactory(
        invitee=UserFactory(is_active=False),
        organization=OrganizationFactory(),
        zones=[zone],
        expires_at=timezone.now(),
    )

    response = UserResendInvitation.mutate(
        root=None,
        info=mock.Mock(
            context=mock.Mock(
                user=requesting_user,
            )
        ),
        invitation_token=invitation.token,
    )

    assert response.status is True

    users = list(User.objects.filter(email=invitation.invitee.email))

    assert len(users) == 1
    assert users[0].is_active is False

    new_invitation = list(
        Invitation.objects.filter(invitee__email=invitation.invitee.email)
    )

    assert len(new_invitation) == 2
    assert new_invitation[0].invitee == invitation.invitee


@pytest.mark.django_db
@mock.patch("core.portal.accounts.graphql.mutations.Auth0ManagementClient")
def test_account_change_name(mock_auth0_management_client) -> None:
    """Test to verify we can change a user's given name.

    Args:
        mock_auth0_management_client (mock.Mock): mocked instance of auth0 client
    """

    requesting_user = UserFactory()

    auth0_id = "auth0|test"
    requesting_user.profile.data = {"auth0_id": auth0_id}
    requesting_user.save()

    requesting_user.permissions = [SELF_UPDATE_PROFILE.global_permission_key]

    response = UserNameUpdate.mutate(
        root=None,
        info=mock.Mock(
            context=mock.Mock(
                user=requesting_user,
            )
        ),
        user_id=auth0_id,
        first_name="John",
    )

    assert response.user == requesting_user

    auth0_client = mock_auth0_management_client()
    auth0_client.users.update.assert_called_once_with(
        auth0_id, {"given_name": "John"}
    )


@pytest.mark.django_db
@mock.patch("core.portal.accounts.graphql.mutations.Auth0ManagementClient")
def test_account_change_mfa_on(
    mock_auth0_management_client: mock.Mock,
) -> None:
    """Test to verify we can change a user's mfa on.

    Args:
        mock_auth0_management_client (mock.Mock): mocked instance of auth0 client
    """
    requesting_user = UserFactory()

    auth0_id = "auth0|test"
    requesting_user.profile.data = {"auth0_id": auth0_id}
    requesting_user.save()

    requesting_user.permissions = [SELF_UPDATE_PROFILE.global_permission_key]

    response = UserMFAUpdate.mutate(
        root=None,
        info=mock.Mock(
            context=mock.Mock(
                user=requesting_user,
            )
        ),
        user_id=auth0_id,
        toggled_mfa_on=True,
    )

    assert response.success is True

    auth0_client = mock_auth0_management_client()
    auth0_client.users.update.assert_called_once_with(
        auth0_id,
        {
            "user_metadata": {
                "has_mfa": True,
            },
        },
    )


@pytest.mark.django_db
@mock.patch("core.portal.accounts.graphql.mutations.Auth0ManagementClient")
def test_account_change_mfa_off(
    mock_auth0_management_client: mock.Mock,
) -> None:
    """Test to verify we can change a user's mfa off.

    Args:
        mock_auth0_management_client (mock.Mock): mocked instance of auth0 client
    """
    requesting_user = UserFactory()

    auth0_id = "auth0|test"
    requesting_user.profile.data = {"auth0_id": auth0_id}
    requesting_user.save()

    requesting_user.permissions = [SELF_UPDATE_PROFILE.global_permission_key]

    response = UserMFAUpdate.mutate(
        root=None,
        info=mock.Mock(
            context=mock.Mock(
                user=requesting_user,
            )
        ),
        user_id=auth0_id,
        toggled_mfa_on=False,
    )

    assert response.success is True

    auth0_client = mock_auth0_management_client()
    auth0_client.users.update.assert_called_once_with(
        auth0_id,
        {
            "user_metadata": {
                "has_mfa": False,
            },
        },
    )
    auth0_client.users.delete_authenticators.assert_called_once_with(auth0_id)


@pytest.mark.django_db
@mock.patch("core.portal.accounts.graphql.mutations.pk_from_global_id")
def test_reviewer_user_role_update(mock_pk_from_global_id):
    user = UserFactory()
    user.profile.data = {"shadow_reviewer": False}

    mock_pk_from_global_id.return_value = (None, user.pk)

    user.permissions = [REVIEWER_ACCOUNTS_UPDATE_ROLE.global_permission_key]

    res = ReviewerAccountRoleUpdate.mutate(
        root=None,
        info=mock.Mock(
            context=mock.Mock(
                user=user,
            )
        ),
        user_id=user.pk,
    )

    updated_user = User.objects.get(pk=user.pk)

    assert isinstance(res, ReviewerAccountRoleUpdate)
    assert updated_user.profile.data["shadow_reviewer"] is True
