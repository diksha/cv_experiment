import graphene
import pytest
from django.contrib.auth.models import User
from graphene.test import Client

from core.portal.accounts import permissions
from core.portal.api.models.organization import Organization
from core.portal.incidents.graphql.schema import IncidentQueries
from core.portal.testing.factories import (
    IncidentFactory,
    OrganizationFactory,
    UserFactory,
    ZoneFactory,
)
from core.portal.testing.graphene_utils import GrapheneContext
from core.portal.zones.models.zone import Zone

# **************************************************************************
# Fixtures
# **************************************************************************


@pytest.fixture(name="graphene_client")
def _graphene_client():
    return Client(graphene.Schema(query=IncidentQueries))


@pytest.fixture(name="organization")
def _organization():
    return OrganizationFactory(key="TEST_ORG", name="Test Org")


@pytest.fixture(name="zone")
def _zone(organization: Organization):
    return ZoneFactory(
        key="TEST_ZONE", name="Test Zone", organization=organization
    )


@pytest.fixture(name="global_user")
def _global_user(organization: Organization, zone: Zone):
    user = UserFactory(
        username="global_user",
        email="global_user@example.com",
        is_staff=True,
        is_superuser=False,
        profile__organization=organization,
        profile__site=zone,
        permissions=[
            permissions.INCIDENT_DETAILS_READ.global_permission_key,
        ],
    )
    organization.users.add(user)
    zone.users.add(user)
    return user


@pytest.fixture(name="organization_user")
def _organization_user(organization: Organization, zone: Zone):
    user = UserFactory(
        username="organization_user",
        email="organization_user@example.com",
        is_staff=True,
        is_superuser=False,
        profile__organization=organization,
        profile__site=zone,
        permissions=[
            permissions.INCIDENT_DETAILS_READ.organization_permission_key,
        ],
    )
    organization.users.add(user)
    zone.users.add(user)
    return user


@pytest.fixture(name="zone_user")
def _zone_user(organization: Organization, zone: Zone):
    user = UserFactory(
        username="zone_user",
        email="zone_user@example.com",
        is_staff=True,
        is_superuser=False,
        profile__organization=organization,
        profile__site=zone,
        permissions=[
            permissions.INCIDENT_DETAILS_READ.zone_permission_key,
        ],
    )
    organization.users.add(user)
    zone.users.add(user)
    return user


@pytest.fixture(name="experimental_user")
def _experimental_user(organization: Organization, zone: Zone):
    user = UserFactory(
        username="global_user",
        email="global_user@example.com",
        is_staff=True,
        is_superuser=False,
        profile__organization=organization,
        profile__site=zone,
        permissions=[
            permissions.EXPERIMENTAL_INCIDENTS_READ.global_permission_key,
        ],
    )
    organization.users.add(user)
    zone.users.add(user)
    return user


# **************************************************************************
# Non-experimental incident tests
# **************************************************************************


@pytest.mark.django_db
def test_resolve_incident_details_global_user_has_access_to_incident_outside_organization(
    graphene_client: Client,
    global_user: User,
) -> None:
    incident = IncidentFactory()
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=global_user),
    )
    assert incident.organization not in global_user.organizations.all()
    assert response["data"]["incidentDetails"]["uuid"] == str(incident.uuid)


@pytest.mark.django_db
def test_resolve_incident_details_global_user_has_access_to_incident_inside_organization(
    graphene_client: Client,
    global_user: User,
) -> None:
    incident = IncidentFactory(organization=global_user.organizations.first())
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=global_user),
    )
    assert incident.organization in global_user.organizations.all()
    assert response["data"]["incidentDetails"]["uuid"] == str(incident.uuid)


@pytest.mark.django_db
def test_resolve_incident_details_unauthorized_user_denied_access_to_incident(
    graphene_client: Client,
) -> None:
    user = UserFactory()
    incident = IncidentFactory()
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=user),
    )
    assert response["data"]["incidentDetails"] is None
    assert response["errors"] is not None


@pytest.mark.django_db
def test_resolve_incident_details_organization_user_has_access_to_incident_inside_organization(
    graphene_client: Client,
    organization_user: User,
    organization: Organization,
    zone: Zone,
) -> None:
    incident = IncidentFactory(
        organization=organization,
        zone=zone,
    )
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=organization_user),
    )
    assert incident.organization in organization_user.organizations.all()
    assert response is not None
    assert response["data"] is not None
    assert response["data"]["incidentDetails"]["uuid"] == str(incident.uuid)


@pytest.mark.django_db
def test_resolve_incident_details_organization_user_denied_access_to_incident_outside_organization(
    graphene_client: Client,
    organization_user: User,
) -> None:
    incident = IncidentFactory(organization=OrganizationFactory())
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=organization_user),
    )
    assert incident.organization not in organization_user.organizations.all()
    assert response["data"]["incidentDetails"] is None
    assert response["errors"] is not None


@pytest.mark.django_db
def test_resolve_incident_details_zone_user_has_access_to_incident_inside_zone(
    graphene_client: Client,
    zone_user: User,
    organization: Organization,
    zone: Zone,
):
    incident = IncidentFactory(
        organization=organization,
        zone=zone,
    )
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=zone_user),
    )
    assert incident.zone in zone_user.zones.all()
    assert response is not None
    assert response["data"] is not None
    assert response["data"]["incidentDetails"]["uuid"] == str(incident.uuid)


@pytest.mark.django_db
def test_resolve_incident_details_zone_user_denied_access_to_incident_outside_zone(
    graphene_client: Client,
    zone_user: User,
    organization: Organization,
) -> None:
    incident = IncidentFactory(organization=organization, zone=ZoneFactory())
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=zone_user),
    )
    assert incident.organization in zone_user.organizations.all()
    assert incident.zone not in zone_user.zones.all()
    assert response["data"]["incidentDetails"] is None
    assert response["errors"] is not None


# **************************************************************************
# Experimental incident tests
# **************************************************************************


@pytest.mark.django_db
def test_resolve_incident_details_authorized_global_user_has_access_to_experimental_incident(
    graphene_client: Client,
    experimental_user: User,
):
    incident = IncidentFactory(
        experimental=True,
    )
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=experimental_user),
    )
    assert response is not None
    assert response["data"] is not None
    assert response["data"]["incidentDetails"]["uuid"] == str(incident.uuid)


@pytest.mark.django_db
def test_resolve_incident_details_unauthorized_global_user_denied_access_to_experimental_incident(
    graphene_client: Client,
    global_user: User,
    organization: Organization,
) -> None:
    incident = IncidentFactory(
        organization=organization,
        zone=ZoneFactory(),
        experimental=True,
    )
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=global_user),
    )
    assert response["data"]["incidentDetails"] is None
    assert response["errors"] is not None


@pytest.mark.django_db
def test_resolve_incident_details_unauthorized_organization_user_denied_access_to_experimental_incident_inside_organization(
    graphene_client: Client,
    organization_user: User,
    organization: Organization,
) -> None:
    incident = IncidentFactory(
        organization=organization,
        zone=ZoneFactory(),
        experimental=True,
    )
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=organization_user),
    )
    assert incident.organization in organization_user.organizations.all()
    assert response["data"]["incidentDetails"] is None
    assert response["errors"] is not None


@pytest.mark.django_db
def test_resolve_incident_details_unauthorized_zone_user_denied_access_to_experimental_incident_inside_zone(
    graphene_client: Client,
    zone_user: User,
    organization: Organization,
    zone: Zone,
) -> None:
    incident = IncidentFactory(
        organization=organization,
        zone=zone,
        experimental=True,
    )
    response = graphene_client.execute(
        f"""{{ incidentDetails(incidentUuid: "{incident.uuid}") {{ uuid }} }}""",
        context=GrapheneContext(user=zone_user),
    )
    assert incident.organization in zone_user.organizations.all()
    assert incident.zone in zone_user.zones.all()
    assert response["data"]["incidentDetails"] is None
    assert response["errors"] is not None
