from datetime import timedelta

import pytest
from django.utils import timezone

from core.portal.demos.jobs.refresh_demo_data import (
    DEMO_ORG_KEY,
    RefreshDemoDataJob,
)
from core.portal.testing.factories import (
    CameraFactory,
    IncidentFactory,
    OrganizationFactory,
    ZoneFactory,
)

INCIDENTS_PER_ORG = 10


@pytest.mark.django_db
def test_refreshes_only_sandbox_org_data() -> None:
    """Test that refresh job only updates demo/sandbox org data."""
    # Original timestamp must be 1+ hour in the past
    original_timestamp = timezone.now() - timedelta(hours=1)
    target_timestamp = original_timestamp + timedelta(hours=1)

    demo_org = OrganizationFactory(key=DEMO_ORG_KEY, is_sandbox=True)
    demo_site = ZoneFactory(key="DEMO_SITE", organization=demo_org)
    demo_camera = CameraFactory(organization=demo_org, zone=demo_site)
    for _ in range(INCIDENTS_PER_ORG):
        IncidentFactory(
            timestamp=original_timestamp,
            organization=demo_org,
            camera=demo_camera,
            zone=demo_site,
        )

    non_demo_org = OrganizationFactory(is_sandbox=False)
    non_demo_site = ZoneFactory(key="NON_DEMO_SITE", organization=non_demo_org)
    non_demo_camera = CameraFactory(
        organization=non_demo_org, zone=non_demo_site
    )
    for _ in range(INCIDENTS_PER_ORG):
        IncidentFactory(
            timestamp=original_timestamp,
            organization=non_demo_org,
            camera=non_demo_camera,
            zone=non_demo_site,
        )

    # Run the job
    RefreshDemoDataJob().run()

    # Demo incidents should have their timestamp updated by 1 hour
    assert demo_org.incidents.count() == INCIDENTS_PER_ORG
    for incident in demo_org.incidents.all():
        assert incident.timestamp == target_timestamp

    # Non-demo incidents should not have their timestamps modified
    assert non_demo_org.incidents.count() == INCIDENTS_PER_ORG
    for incident in non_demo_org.incidents.all():
        assert incident.timestamp == original_timestamp
