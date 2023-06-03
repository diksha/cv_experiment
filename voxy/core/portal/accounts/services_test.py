import mock
import pytest

from core.portal.accounts.services import register_user
from core.portal.api.models.invitation import Invitation
from core.portal.testing.factories import (
    InvitationFactory,
    OrganizationFactory,
    UserFactory,
    ZoneFactory,
)


@pytest.mark.django_db
def test_register_user_happy_path():
    """Verifies register_user function has the expected side effects."""
    mock_auth0_client = mock.Mock(
        users=mock.Mock(create=mock.Mock(return_value={"user_id": "test_id"}))
    )

    invitation = InvitationFactory(
        invitee=UserFactory(is_active=False),
        organization=OrganizationFactory(),
        zones=[ZoneFactory(), ZoneFactory()],
    )
    register_user(
        mock_auth0_client, invitation, "testfirst", "testlast", "testpassword"
    )

    updated_invitation = Invitation.objects.get(pk=invitation.pk)
    assert updated_invitation.redeemed is True

    new_user = updated_invitation.invitee
    assert new_user.is_active is True
    assert new_user.profile.data["auth0_id"] == "test_id"
    assert new_user.profile.site in invitation.zones.all()
    assert new_user.profile.organization == invitation.organization
    assert new_user in invitation.organization.users.all()
    for zone in invitation.zones.all():
        assert new_user in zone.users.all()
