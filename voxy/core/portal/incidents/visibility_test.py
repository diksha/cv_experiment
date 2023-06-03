import pytest

from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.incidents.visibility import (
    maybe_hide_incident_from_customers,
    maybe_show_incident_to_customers,
)
from core.portal.notifications.enums import NotificationCategory
from core.portal.testing.factories import (
    IncidentFactory,
    NotificationLogFactory,
)


@pytest.mark.django_db
def test_maybe_show_incident_to_customers() -> None:
    incident = IncidentFactory(
        review_level=ReviewLevel.GREEN,
        visible_to_customers=False,
    )
    maybe_show_incident_to_customers(incident)
    assert incident.visible_to_customers is True


@pytest.mark.django_db
def test_maybe_hide_incident_from_customers_when_no_alerts_exist() -> None:
    incident = IncidentFactory(
        review_level=ReviewLevel.GREEN,
        visible_to_customers=True,
    )
    maybe_hide_incident_from_customers(incident)
    assert incident.visible_to_customers is False


@pytest.mark.django_db
def test_maybe_hide_incident_from_customers_when_alerts_exist() -> None:
    incident = IncidentFactory(
        review_level=ReviewLevel.GREEN,
        visible_to_customers=True,
    )
    NotificationLogFactory(
        incident=incident, category=NotificationCategory.INCIDENT_ALERT
    )
    maybe_hide_incident_from_customers(incident)
    assert incident.visible_to_customers is True
