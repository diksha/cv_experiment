from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from typing import Dict, List, Optional, Tuple

from django.conf import settings
from django.contrib.auth.models import User
from django.utils import timezone
from loguru import logger

from core.portal.api.models.incident import Incident
from core.portal.api.models.notification_log import NotificationLog
from core.portal.incidents.enums import IncidentTypeKey
from core.portal.notifications.clients.sendgrid import (
    INCIDENT_ALERT_ASM_GROUP_ID,
    INCIDENT_ALERT_TEMPLATE_ID,
    SendGridClient,
)
from core.portal.notifications.enums import NotificationCategory


@dataclass
class AlertConfig:
    min_hours_since_last_alert: int
    enabled: bool


alert_configs: Dict[Tuple[str, str, str], AlertConfig] = {
    (
        "USCOLD",
        "LAREDO",
        IncidentTypeKey.SPILL,
    ): AlertConfig(min_hours_since_last_alert=8, enabled=True),
    (
        "PPG",
        "CEDAR_FALLS",
        IncidentTypeKey.SPILL,
    ): AlertConfig(min_hours_since_last_alert=8, enabled=True),
    (
        "PPG",
        "CEDAR_FALLS",
        IncidentTypeKey.NO_STOP_AT_END_OF_AISLE,
    ): AlertConfig(min_hours_since_last_alert=1, enabled=True),
    (
        "WESCO",
        "RENO",
        IncidentTypeKey.NO_STOP_AT_END_OF_AISLE,
    ): AlertConfig(min_hours_since_last_alert=0, enabled=True),
    (
        "WESCO",
        "RENO",
        IncidentTypeKey.SAFETY_VEST,
    ): AlertConfig(min_hours_since_last_alert=0, enabled=False),
    (
        "WESCO",
        "RENO",
        IncidentTypeKey.BAD_POSTURE,
    ): AlertConfig(min_hours_since_last_alert=0, enabled=True),
}


class AlertManager:
    """Responsible for sending incident alerts."""

    def __init__(self, incident: Incident) -> None:
        self.incident = incident
        self.organization = incident.organization
        self.zone = incident.zone
        self.incident_type = incident.incident_type

    @cached_property
    def _valid_incident(self) -> bool:
        return (
            self.incident
            and self.organization
            and self.zone
            and self.incident_type
        )

    @cached_property
    def _incident_type_name(self) -> str:
        org_incident_type = (
            self.incident_type.organization_incident_types.filter(
                organization=self.organization,
            ).first()
        )
        if org_incident_type:
            return org_incident_type.name
        return self.incident_type.name

    @cached_property
    def _alert_config(self) -> Optional[AlertConfig]:
        config_key = (
            self.organization.key,
            self.zone.key,
            self.incident_type.key,
        )
        return alert_configs.get(config_key, None)

    @cached_property
    def _recent_alert_sent(self) -> bool:
        time_since_last_alert_threshold = timezone.now() - timedelta(
            hours=self._alert_config.min_hours_since_last_alert
        )
        return NotificationLog.objects.filter(
            category=NotificationCategory.INCIDENT_ALERT,
            sent_at__gte=time_since_last_alert_threshold,
            incident__organization=self.organization,
            incident__zone=self.zone,
            incident__incident_type=self.incident_type,
        ).exists()

    @cached_property
    def _recipients(self) -> List[User]:
        return list(
            self.zone.active_users.filter(
                profile__data__receive_incident_alerts=True,
            )
        )

    def maybe_send_alert(self) -> None:
        if not self._valid_incident:
            logger.warning(
                f"Not sending alert for invalid incident (UUID: {self.incident.uuid})"
            )
            return

        if not self.incident.visible_to_customers or not self._alert_config:
            return

        if self._recent_alert_sent:
            logger.warning(
                f"Not sending alert for incident (UUID: {self.incident.uuid}) because"
                " alerts have been sent for this incident type within the past"
                " {config.min_hours_since_last_alert} hour(s)"
            )
            return

        incident_url = f"{settings.BASE_URL}/incidents/{self.incident.uuid}?REDIRECT_SITE_KEY={self.zone.key}"
        incident_timestamp_formatted = self.incident.timestamp.astimezone(
            self.zone.tzinfo
        ).strftime("%A, %B %-d @ %-I:%M%p %Z")
        subject = (
            f"Incident Alert: {self._incident_type_name} at {self.zone.name}"
        )

        def send_email(to_email: str) -> None:
            SendGridClient().send_email_with_template(
                from_email=settings.DEFAULT_FROM_EMAIL,
                to_emails=[to_email],
                subject=subject,
                template_id=INCIDENT_ALERT_TEMPLATE_ID,
                asm_group_id=INCIDENT_ALERT_ASM_GROUP_ID,
                incident_url=incident_url,
                incident_type_name=self._incident_type_name,
                location_name=self.zone.name,
                camera_name=self.incident.camera.name,
                incident_timestamp_formatted=incident_timestamp_formatted,
            )

        for recipient in self._recipients:
            send_email(recipient.email)
            NotificationLog.objects.create(
                user=recipient,
                sent_at=timezone.now(),
                incident=self.incident,
                category=NotificationCategory.INCIDENT_ALERT,
            )

            logger.info(
                f"Incident alert (UUID: {self.incident.uuid}) sent to {recipient.email}"
            )

        # Send a copy to internal google group
        send_email(settings.CUSTOMER_ALERT_EMAILS_BCC)

        # Flag the incident as alerted
        self.incident.alerted = True
        self.incident.save()
