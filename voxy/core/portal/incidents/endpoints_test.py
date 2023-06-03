import mock
import pytest
from django.contrib.auth import get_user_model
from django.test import Client

from core.portal.api.models.incident import Incident
from core.portal.api.models.incident_type import (
    IncidentType,
    OrganizationIncidentType,
)
from core.portal.api.models.organization import Organization
from core.portal.api.models.share_link import ShareLink
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.lib.utils.signed_url_manager import signed_url_manager
from core.portal.testing.factories import (
    IncidentTypeFactory,
    OrganizationFactory,
    ShareLinkFactory,
    UserFactory,
)

User = get_user_model()


@pytest.fixture(name="user")
def _user():
    """User fixture.

    Returns:
        User: user
    """
    return UserFactory(
        username="admin",
        email="admin1@example.com",
        is_staff=True,
        is_superuser=True,
    )


@pytest.fixture(name="bad_posture")
def _bad_posture():
    """Bad posture incident type fixture

    Returns:
        IncidentType: bad posture incident type
    """
    return IncidentTypeFactory(
        key="BAD_POSTURE",
        name="Bad Posture",
    )


@pytest.fixture(name="organization")
def _organization(bad_posture: IncidentType):
    """Organization fixture

    Args:
        bad_posture (IncidentType): enabled incident type

    Returns:
        Organization: organization
    """
    return OrganizationFactory(key="ACME", incident_types=[bad_posture])


@pytest.mark.django_db
def test_default_review_level(
    client: Client,
    user: User,
    organization: Organization,
    bad_posture: IncidentType,
) -> None:
    """Test that created incidents get default review level of red.

    Args:
        client (Client): django test client
        user (User): user issuing request
        organization (Organization): created incident's organization
        bad_posture (IncidentType): created incident's type
    """
    client.force_login(user)
    client.post(
        "/api/incidents/",
        {
            "organization_key": organization.key,
            "incident_type_id": bad_posture.key,
            "start_frame_relative_ms": 1.5,
            "uuid": "5b4a8a07-540f-46b9-93fc-04bb7cc040c5",
            "title": "Test title",
            "priority": "1",
            "incident_version": "1",
        },
    )
    incident = Incident.objects.get(
        uuid="5b4a8a07-540f-46b9-93fc-04bb7cc040c5"
    )
    assert incident.review_level == ReviewLevel.RED


@pytest.mark.django_db
def test_organization_custom_review_level(
    client: Client,
    user: User,
    organization: Organization,
    bad_posture: IncidentType,
) -> None:
    """Test that created incidents get organization's custom review level.

    Args:
        client (Client): django test client
        user (User): user issuing request
        organization (Organization): created incident's organization
        bad_posture (IncidentType): created incident's type
    """
    # Set review level to GREEN
    org_incident_type = OrganizationIncidentType.objects.get(
        organization=organization, incident_type=bad_posture
    )
    org_incident_type.review_level = ReviewLevel.GREEN
    org_incident_type.save()

    client.force_login(user)
    client.post(
        "/api/incidents/",
        {
            "organization_key": organization.key,
            "incident_type_id": bad_posture.key,
            "start_frame_relative_ms": 1.5,
            "uuid": "bb4a8a07-540f-46b9-93fc-04bb7cc040c5",
            "title": "Test title",
            "priority": "1",
            "incident_version": "1",
        },
    )
    incident = Incident.objects.get(
        uuid="bb4a8a07-540f-46b9-93fc-04bb7cc040c5"
    )
    assert incident.review_level == ReviewLevel.GREEN


@pytest.mark.django_db
def test_redeem_share_link(
    client: Client,
) -> None:
    """Test that incident share link is redeemed correctly.

    Args:
        client (Client): django test client

    """
    with mock.patch.object(
        signed_url_manager, "get_signed_url", return_value="test_video_url"
    ):
        share_link = ShareLinkFactory()

        token = share_link.token
        incident = share_link.incident

        assert share_link.visits == 0

        response = client.get(f"/api/share/{token}/")

        assert response.status_code == 200

        expected_data = {
            "title": incident.title,
            "id": incident.id,
            "zone_name": incident.zone.name,
            "video_url": "test_video_url",
            "camera_name": incident.camera.name,
            "timestamp": incident.timestamp,
            "annotations_url": incident.annotations_url,
            "actor_ids": incident.actor_ids,
        }

        assert response.data == expected_data

        post_request_share_link = ShareLink.objects.get(
            token=token,
        )

        assert post_request_share_link.visits == 1
