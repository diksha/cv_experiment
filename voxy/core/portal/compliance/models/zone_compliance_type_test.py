import pytest

from core.portal.testing.factories import (
    ComplianceTypeFactory,
    ZoneComplianceTypeFactory,
)


@pytest.mark.django_db
def test_name_property_overrides_base_name_value() -> None:
    """Verifies .name property's override behavior"""
    compliance_type = ComplianceTypeFactory(name="Old")
    zone_compliance_type = ZoneComplianceTypeFactory(
        compliance_type=compliance_type, name_override="New"
    )
    assert zone_compliance_type.name == "New"
