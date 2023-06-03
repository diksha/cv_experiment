from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import cached_property
from typing import Dict, List

from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Count, Max
from django.utils import timezone
from loguru import logger

from core.portal.api.models.notification_log import NotificationLog
from core.portal.api.models.organization import Organization
from core.portal.lib.utils.date_utils import hours_between
from core.portal.notifications.clients.sendgrid import (
    ZONE_PULSE_ASM_GROUP_ID,
    ZONE_PULSE_TEMPLATE_ID,
    SendGridClient,
)
from core.portal.notifications.enums import NotificationCategory
from core.portal.notifications.jobs.base import ScheduledNotificationJob
from core.portal.zones.models.zone import Zone


@dataclass(frozen=True)
class Trigger:
    """When a pulse notification should be triggered."""

    hour: int
    minute: int


@dataclass(frozen=True)
class PulseConfig:
    """What data should be included in a pulse notification."""

    timespan_desc: str
    delay_minutes: int
    hours_covered: int
    weekdays_active: List[int]


ConfigMapType = Dict[str, Dict[Trigger, PulseConfig]]

config_map: ConfigMapType = {
    "LAREDO": {
        # Working hours (CST):
        #     M-F: 5am - 2am
        #     Sat: 5am - 2pm
        #     Sun: Closed
        Trigger(2, 30): PulseConfig("11pm - 2am", 30, 3, [1, 2, 3, 4, 5]),
        Trigger(7, 30): PulseConfig("5am - 7am", 30, 2, [0, 1, 2, 3, 4, 5]),
        Trigger(9, 30): PulseConfig("7am - 9am", 30, 2, [0, 1, 2, 3, 4, 5]),
        Trigger(11, 30): PulseConfig("9am - 11am", 30, 2, [0, 1, 2, 3, 4, 5]),
        Trigger(13, 30): PulseConfig("11am - 1pm", 30, 2, [0, 1, 2, 3, 4, 5]),
        Trigger(15, 30): PulseConfig("1pm - 3pm", 30, 2, [0, 1, 2, 3, 4, 5]),
        Trigger(17, 30): PulseConfig("3pm - 5pm", 30, 2, [0, 1, 2, 3, 4]),
        Trigger(19, 30): PulseConfig("5pm - 7pm", 30, 2, [0, 1, 2, 3, 4]),
        Trigger(21, 30): PulseConfig("7pm - 9pm", 30, 2, [0, 1, 2, 3, 4]),
        Trigger(23, 30): PulseConfig("9pm - 11pm", 30, 2, [0, 1, 2, 3, 4]),
    }
}


@dataclass
class IncidentTypeCount:
    key: str
    name: str
    count: int


@dataclass
class NotificationData:
    notification_subject: str
    timespan_desc: str
    organization_name: str
    zone_name: str
    incident_type_counts: List[IncidentTypeCount]
    dashboard_url: str


class ZonePulseJob(ScheduledNotificationJob):
    """Hacky solution for sending periodic "pulse" emails.

    These are called "pulse" emails because they'll be sent out every N hours
    throughout the day. The customer value is that recipients can get a high
    level overview of what's going on at their site throughout the day without
    needing to open the Voxel app.

    This job will be invoked by Google Cloud Scheduler twice per hour:

        :00
        :30

    This implementation depends on being invoked at exactly these times
    and doesn't have any logic to handle delays. If it is invoked at :31
    then there will be no matching config and no emails will be sent. This
    should be fine as Cloud Scheduler is quite reliable and if customers
    miss an email or two it's not critical...but it is a risk.

    Again, this is a hack so it's not intended to be scalable or flexible
    for many use cases. Ideally we can delete this in the very near future,
    so please don't extend or modify this if at all possible.
    """

    def __init__(
        self,
        zone_id: int,
        timespan_desc: str,
        invocation_timestamp: datetime,
        localized_start_timestamp: datetime,
        localized_end_timestamp: datetime,
        base_url: str,
    ):
        self.zone_id = zone_id
        self.timespan_desc = timespan_desc
        self.invocation_timestamp = invocation_timestamp
        self.localized_start_timestamp = localized_start_timestamp
        self.localized_end_timestamp = localized_end_timestamp
        self.base_url = base_url

    @classmethod
    def get_jobs_to_run(
        cls,
        invocation_timestamp: datetime = None,
        base_url: str = settings.BASE_URL,
    ) -> List["ZonePulseJob"]:
        """Factory function which returns list of jobs to be run.

        1. Invoke every 30m via external scheduler (assume this is reliable)
        2. Get zones with a pulse trigger configured for the invocation timestamp
        3. Filter out zones where the pulse trigger is not enabled for the current weekday
        4. For each zone, calculate the start/end timestamps for the current pulse
           and add a job to the return list

        Args:
            invocation_timestamp (datetime): timestamp of when function was invoked.
            base_url (str): Base URL to use in content links.

        Returns:
            List of ready-to-run job instances.
        """

        if not invocation_timestamp:
            invocation_timestamp = timezone.now()

        logger.info("Getting jobs to run for zone pulse notifications")

        jobs_to_run = []
        for zone_key, zone_config in config_map.items():
            zone = Zone.objects.get(key=zone_key)
            localized_invocation_timestamp = invocation_timestamp.astimezone(
                zone.tzinfo
            )
            current_hour = localized_invocation_timestamp.hour
            current_minute = localized_invocation_timestamp.minute
            current_weekday = localized_invocation_timestamp.weekday()
            config = zone_config.get(Trigger(current_hour, current_minute))

            if config and current_weekday in config.weekdays_active:
                # Calculate start/end timestamps for the current pulse time period
                localized_start_timestamp = (
                    localized_invocation_timestamp
                    - timedelta(
                        hours=config.hours_covered,
                        minutes=config.delay_minutes,
                    )
                ).replace(minute=0, second=0, microsecond=0)
                localized_end_timestamp = (
                    localized_start_timestamp
                    + timedelta(hours=config.hours_covered)
                    - timedelta(microseconds=1)
                )
                jobs_to_run.append(
                    ZonePulseJob(
                        zone_id=zone.id,
                        timespan_desc=config.timespan_desc,
                        invocation_timestamp=invocation_timestamp,
                        localized_start_timestamp=localized_start_timestamp,
                        localized_end_timestamp=localized_end_timestamp,
                        base_url=base_url,
                    )
                )
                logger.info(
                    f"Queued job to run for zone ID {zone.id} ({config.timespan_desc}, {localized_start_timestamp} - {localized_end_timestamp})"
                )
        return jobs_to_run

    @property
    def notification_category(self) -> NotificationCategory:
        return NotificationCategory.ZONE_PULSE

    @cached_property
    def zone(self) -> Zone:
        return Zone.objects.get(pk=self.zone_id)

    @cached_property
    def organization(self) -> Organization:
        return self.zone.organization

    def get_recipients(self) -> List[User]:
        """List of users who should receive notifications for this job."""

        preference_key = "receive_pulse_emails"
        filter_key = f"profile__data__{preference_key}"
        recipients = list(self.zone.active_users.filter(**{filter_key: True}))

        recipient_map: Dict[int, User] = {r.id: r for r in recipients}

        last_sent_values = (
            NotificationLog.objects.filter(
                user__in=recipients,
                category=self.notification_category,
            )
            .annotate(last_sent=Max("sent_at"))
            .values("user_id", "sent_at")
        )

        for record in last_sent_values:
            last_notification = record.get("sent_at")
            if last_notification:
                hours_since_last_notification = hours_between(
                    last_notification, self.invocation_timestamp
                )
                if hours_since_last_notification <= 1.0:
                    # Don't send emails if this user received one within the last 1 hour
                    user_id = record.get("user_id")
                    logger.warning(
                        f"Not sending pulse notification to user ID {user_id}"
                    )
                    recipient_map.pop(user_id)

        return list(recipient_map.values())

    def get_data(self) -> NotificationData:
        """Gets notification job data."""

        INCIDENT_TYPE_KEY = "incident_type__key"
        COUNT = "count"

        incident_type_counts_queryset = (
            self.zone.incidents.filter(
                visible_to_customers=True,
                timestamp__gte=self.localized_start_timestamp,
                timestamp__lte=self.localized_end_timestamp,
            )
            .values(INCIDENT_TYPE_KEY)
            .annotate(**{COUNT: Count(INCIDENT_TYPE_KEY)})
            .order_by("count")
        )

        # Names may be overridden at the org level
        org_incident_type_names = {
            eit.key: eit.name
            for eit in self.organization.enabled_incident_types
        }

        incident_type_counts: List[Dict[str, int]] = []
        for row in incident_type_counts_queryset:
            key = row.get(INCIDENT_TYPE_KEY)
            name = org_incident_type_names.get(key, "Unknown")
            count = row.get(COUNT, 0)
            incident_type_counts.append(IncidentTypeCount(key, name, count))

        # Wednesday, Dec 8
        date_string = self.localized_start_timestamp.strftime("%A, %b %-d")

        data = NotificationData(
            notification_subject=f"Pulse for {date_string}",
            timespan_desc=self.timespan_desc,
            organization_name=self.organization.name,
            zone_name=self.zone.name,
            incident_type_counts=incident_type_counts,
            dashboard_url=self.base_url,
        )
        return data

    def send_email(self, to_emails: List[str], data: NotificationData):
        SendGridClient().send_email_with_template(
            from_email=settings.DEFAULT_FROM_EMAIL,
            to_emails=to_emails,
            subject=data.notification_subject,
            template_id=ZONE_PULSE_TEMPLATE_ID,
            asm_group_id=ZONE_PULSE_ASM_GROUP_ID,
            **asdict(data),
        )
        logger.info(f"Pulse email for {self.zone.key} sent to {to_emails}")

    def run(self) -> None:
        """Runs the job."""
        data = self.get_data()
        recipients = self.get_recipients()
        for recipient in recipients:
            self.send_email(to_emails=[recipient.email], data=data)
            NotificationLog.objects.create(
                user=recipient,
                sent_at=timezone.now(),
                category=self.notification_category,
            )

        # Send to google group so we can monitor
        self.send_email(
            to_emails=[settings.CUSTOMER_PULSE_EMAILS_BCC], data=data
        )
