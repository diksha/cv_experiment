import pytest

from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.testing.factories import (
    IncidentTypeFactory,
    OrganizationIncidentTypeFactory,
)


@pytest.mark.django_db
def test_org_incident_type_name_inherits_from_incident_type() -> None:
    """Test that org incident type name inherits from incident type."""
    incident_type = IncidentTypeFactory(key="foo", name="Foo")
    org_type_1 = OrganizationIncidentTypeFactory(
        incident_type=incident_type,
    )
    org_type_2 = OrganizationIncidentTypeFactory(
        incident_type=incident_type,
        name_override="",
    )
    assert org_type_1.name == "Foo"
    assert org_type_2.name == "Foo"


@pytest.mark.django_db
def test_org_incident_type_name_overrides_incident_type() -> None:
    """Test that org incident type name overrides incident type."""
    incident_type = IncidentTypeFactory(key="foo", name="Foo")
    org_incident_type = OrganizationIncidentTypeFactory(
        name_override="Bar",
        incident_type=incident_type,
    )
    assert org_incident_type.name == "Bar"


@pytest.mark.django_db
def test_org_incident_type_review_level_defaults_to_red() -> None:
    """Test that org incident type level defaults to red."""
    incident_type = IncidentTypeFactory(key="foo", name="Foo")
    org_incident_type = OrganizationIncidentTypeFactory(
        name_override="Bar",
        incident_type=incident_type,
    )
    assert org_incident_type.review_level == ReviewLevel.RED
