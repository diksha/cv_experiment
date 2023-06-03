from datetime import datetime

import pytest
from django.utils import timezone
from testing.factories import (
    CameraFactory,
    IncidentTypeFactory,
    OrganizationFactory,
    OrganizationIncidentTypeFactory,
    ZoneFactory,
)

from core.portal.api.models.incident_type import IncidentType
from core.portal.api.models.organization import Organization
from core.portal.devices.models.camera import Camera
from core.portal.incidents.commands import CreateIncident
from core.portal.incidents.enums import CooldownSource
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.zones.models.zone import Zone
from core.structs.incident import Incident as IncidentStruct


@pytest.fixture(name="organization")
def _organization() -> Organization:
    return OrganizationFactory()


@pytest.fixture(name="site")
def _site(organization: Organization) -> Zone:
    return ZoneFactory(organization=organization)


@pytest.fixture(name="incident_type")
def _incident_type() -> IncidentType:
    return IncidentTypeFactory()


@pytest.fixture(name="camera")
def _camera(organization: Organization, site: Zone) -> Camera:
    return CameraFactory(organization=organization, zone=site)


@pytest.fixture(name="timestamp")
def _timestamp() -> datetime:
    # arbitrary timestamp
    return datetime(2023, 3, 22, 1, 34, 23, tzinfo=timezone.utc)


@pytest.fixture(name="valid_data")
def _valid_data(
    timestamp: datetime,
    organization: Organization,
    incident_type: IncidentType,
    camera: Camera,
) -> IncidentStruct:
    return IncidentStruct.from_dict(
        {
            "title": "Test Incident",
            "incident_version": "1.0.0",
            "organization_key": organization.key,
            "incident_type_id": incident_type.key,
            "start_frame_relative_ms": timestamp.timestamp() * 1000,
            "camera_uuid": str(camera.uuid),
            "cooldown_tag": "False",
        }
    )


@pytest.mark.django_db
def test_execute_fails_with_empty_data() -> None:
    empty_data = {}

    with pytest.raises(Exception):
        CreateIncident(empty_data).execute()


@pytest.mark.django_db
def test_execute_creates_incident(
    valid_data: IncidentStruct,
    timestamp: datetime,
    organization: Organization,
    site: Zone,
    incident_type: IncidentType,
    camera: Camera,
) -> None:
    incident = CreateIncident(valid_data).execute()

    assert incident is not None
    assert incident.experimental is False
    assert incident.timestamp == timestamp
    assert incident.organization == organization
    assert incident.zone == site
    assert incident.incident_type == incident_type
    assert incident.camera == camera
    assert incident.cooldown_source is None


@pytest.mark.django_db
def test_execute_creates_incident_with_org_incident_type_review_level(
    valid_data: IncidentStruct,
    organization: Organization,
    incident_type: IncidentType,
) -> None:
    org_incident_type = OrganizationIncidentTypeFactory(
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.GOLD,
    )
    incident = CreateIncident(valid_data).execute()

    assert incident is not None
    assert incident.review_level == org_incident_type.review_level


@pytest.mark.django_db
def test_execute_creates_incident_with_default_red_review_level(
    valid_data: IncidentStruct,
) -> None:
    incident = CreateIncident(valid_data).execute()

    assert incident is not None
    assert incident.review_level == ReviewLevel.RED


@pytest.mark.django_db
def test_execute_creates_experimental_incident_when_data_contains_experimental_version(
    valid_data: IncidentStruct,
) -> None:
    valid_data.incident_version = "experimental-1.0.0"
    incident = CreateIncident(valid_data).execute()

    assert incident is not None
    assert incident.experimental is True


@pytest.mark.django_db
def test_execute_sets_cooldown_source_when_cooldown_tag_present(
    valid_data: IncidentStruct,
) -> None:
    valid_data.cooldown_tag = "True"
    incident = CreateIncident(valid_data).execute()

    assert incident is not None
    assert incident.is_cooldown is True
    assert incident.cooldown_source == CooldownSource.COOLDOWN


@pytest.mark.django_db
def test_new_red_incidents_are_hidden(
    valid_data: IncidentStruct,
    organization: Organization,
    incident_type: IncidentType,
) -> None:
    OrganizationIncidentTypeFactory(
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.RED,
    )
    incident = CreateIncident(valid_data).execute()
    assert incident.visible_to_customers is False


@pytest.mark.django_db
def test_new_yellow_incidents_are_hidden(
    valid_data: IncidentStruct,
    organization: Organization,
    incident_type: IncidentType,
) -> None:
    OrganizationIncidentTypeFactory(
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.YELLOW,
    )
    incident = CreateIncident(valid_data).execute()
    assert incident.visible_to_customers is False


@pytest.mark.django_db
def test_new_gold_incidents_are_visible(
    valid_data: IncidentStruct,
    organization: Organization,
    incident_type: IncidentType,
) -> None:
    OrganizationIncidentTypeFactory(
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.GOLD,
    )
    incident = CreateIncident(valid_data).execute()
    assert incident.visible_to_customers is True


@pytest.mark.django_db
def test_new_green_incidents_are_visible(
    valid_data: IncidentStruct,
    organization: Organization,
    incident_type: IncidentType,
) -> None:
    OrganizationIncidentTypeFactory(
        organization=organization,
        incident_type=incident_type,
        review_level=ReviewLevel.GREEN,
    )
    incident = CreateIncident(valid_data).execute()
    assert incident.visible_to_customers is True
