from datetime import timedelta
from unittest import mock

import pytest
from django.utils import timezone

from core.portal.api.models.notification_log import NotificationLog
from core.portal.incidents.alerts import AlertConfig, AlertManager
from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.notifications.clients.sendgrid import SendGridClient
from core.portal.notifications.enums import NotificationCategory
from core.portal.testing.factories import (
    IncidentFactory,
    NotificationLogFactory,
    UserFactory,
)


@pytest.mark.django_db
@mock.patch.object(SendGridClient, "send_email_with_template")
def test_alert_manager_skips_hidden_incidents(
    mock_send_email_with_template,
) -> None:
    incident = IncidentFactory(
        visible_to_customers=False,
        review_level=ReviewLevel.RED,
    )
    AlertManager(incident).maybe_send_alert()
    assert mock_send_email_with_template.call_count == 0
    assert incident.alerted is False


@pytest.mark.django_db
@mock.patch.object(
    AlertManager,
    "_alert_config",
    new_callable=mock.PropertyMock,
    return_value=AlertConfig(min_hours_since_last_alert=8, enabled=True),
)
@mock.patch.object(SendGridClient, "send_email_with_template")
@mock.patch.object(SendGridClient, "__init__", return_value=None)
def test_alert_manager_sends_email_for_eligible_incident(
    mock_sendgrid_init, mock_send_email_with_template, mock_alert_config
) -> None:
    del mock_alert_config, mock_sendgrid_init
    incident = IncidentFactory(
        visible_to_customers=True,
        review_level=ReviewLevel.GREEN,
    )
    AlertManager(incident).maybe_send_alert()
    assert mock_send_email_with_template.call_count == 1
    assert incident.alerted is True


@pytest.mark.django_db
@mock.patch.object(
    AlertManager,
    "_alert_config",
    new_callable=mock.PropertyMock,
    return_value=AlertConfig(min_hours_since_last_alert=8, enabled=True),
)
@mock.patch.object(SendGridClient, "send_email_with_template")
def test_alert_manager_skips_when_recent_alerts_present(
    mock_send_email_with_template, mock_alert_config
) -> None:

    del mock_alert_config
    incident = IncidentFactory(
        visible_to_customers=True, review_level=ReviewLevel.GREEN
    )
    NotificationLogFactory(
        incident=incident,
        category=NotificationCategory.INCIDENT_ALERT,
        sent_at=timezone.now() - timedelta(hours=1),
    )
    AlertManager(incident).maybe_send_alert()
    assert mock_send_email_with_template.call_count == 0


@pytest.mark.django_db
@mock.patch.object(
    AlertManager,
    "_alert_config",
    new_callable=mock.PropertyMock,
    return_value=AlertConfig(min_hours_since_last_alert=8, enabled=True),
)
@mock.patch.object(SendGridClient, "send_email_with_template")
@mock.patch.object(SendGridClient, "__init__", return_value=None)
def test_alert_manager_logs_activity(
    mock_sendgrid_init, mock_send_email_with_template, mock_alert_config
) -> None:
    del mock_alert_config, mock_sendgrid_init
    incident = IncidentFactory(
        visible_to_customers=True, review_level=ReviewLevel.GREEN
    )

    # Add recipient to zone
    recipient = UserFactory()
    recipient.profile.data = {"receive_incident_alerts": True}
    recipient.profile.save()
    incident.zone.users.add(recipient)

    # Exercise the alert manager
    AlertManager(incident).maybe_send_alert()

    # Emails will be sent to recipient and internal google group
    assert mock_send_email_with_template.call_count == 2

    # Log entry is only added for user recipients, not internal google groups
    log_entries = NotificationLog.objects.filter(
        incident=incident, category=NotificationCategory.INCIDENT_ALERT
    )
    assert log_entries.count() == 1
