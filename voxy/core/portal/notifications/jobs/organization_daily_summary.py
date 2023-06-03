import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any, Dict, FrozenSet, List, Union
from urllib.parse import urlencode

import pytz
from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Max
from django.utils import timezone
from loguru import logger

from core.portal.analytics.services import get_series
from core.portal.api.models.notification_log import NotificationLog
from core.portal.api.models.organization import Organization
from core.portal.lib.utils.date_utils import hours_between
from core.portal.notifications.clients.sendgrid import (
    DAILY_SUMMARY_ASM_GROUP_ID,
    DAILY_SUMMARY_TEMPLATE_ID,
    SendGridClient,
)
from core.portal.notifications.enums import NotificationCategory
from core.portal.notifications.jobs.base import ScheduledNotificationJob
from core.portal.zones.enums import ZoneType
from core.portal.zones.models.zone import Zone


def format_param_value(value: Union[str, List[str]]) -> str:
    """Converts Python value to JSON string for use in query string params."""
    return json.dumps(value)


@dataclass
class SummaryData:
    site_name: str
    high_count: int
    high_count_url: str
    medium_count: int
    medium_count_url: str
    low_count: int
    low_count_url: str
    incident_counts: List[Dict[str, Union[int, str]]]
    highlighted_string: str
    highlighted_thumbnail: str
    highlighted_url: str
    dashboard_url: str = field(default="https://app.voxelai.com")


@dataclass
class SummaryDataAggregate:
    notification_subject: str
    organization_name: str
    sites: List[SummaryData]


class OrganizationDailySummaryJob(ScheduledNotificationJob):
    """Organization daily summary job.

    Runs at 9:00am local time every day and contains a summary of all
    organization activity from the previous day.
    """

    SEND_AT_LOCAL_HOUR = 9

    def __init__(
        self,
        organization_id: int,
        invocation_timestamp: datetime,
        base_url: str,
    ):
        self.organization_id = organization_id
        self.invocation_timestamp = invocation_timestamp
        self.base_url = base_url

    @property
    def notification_category(self) -> NotificationCategory:
        return NotificationCategory.ORGANIZATION_DAILY_SUMMARY

    @cached_property
    def organization(self) -> Organization:
        return Organization.objects.get(pk=self.organization_id)

    @cached_property
    def recipients(self) -> List[User]:
        """List of users who should receive notifications for this job."""
        # TODO: get rid of hardcoded preference key
        recipients = list(
            self.organization.active_users.filter(
                profile__data__receive_daily_summary_emails=True,
            )
        )

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
                    last_notification, self.localized_invocation_timestamp
                )
                if hours_since_last_notification <= 3.0:
                    # Don't send emails if this user received one within the last 3 hours
                    recipient_map.pop(record.get("user_id", None))

        # HACK - always include superusers for monitoring purposes until feature matures
        superusers = User.objects.filter(
            is_superuser=True,
            profile__data__receive_daily_summary_emails=True,
            is_active=True,
        )
        for user in superusers:
            recipient_map[user.id] = user

        return list(recipient_map.values())

    @property
    def localized_invocation_timestamp(self) -> datetime:
        return self.invocation_timestamp.astimezone(
            pytz.timezone(self.organization.timezone)
        )

    @property
    def localized_previous_day(self) -> datetime:
        """Localized previous day."""
        return self.localized_invocation_timestamp - timedelta(days=1)

    @property
    def localized_start(self) -> datetime:
        """Localized start of previous day."""
        return self.localized_previous_day.replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    @property
    def localized_end(self) -> datetime:
        """Localized end of previous day."""
        return self.localized_previous_day.replace(
            hour=23,
            minute=59,
            second=59,
            microsecond=999999,
        )

    @property
    def notification_subject(self) -> str:
        # Daily summary for Wednesday, Dec 8
        date_string = self.localized_previous_day.strftime("%A, %b %-d")
        return f"Daily summary for {date_string}"

    @classmethod
    def get_jobs_to_run(
        cls,
        invocation_timestamp: datetime = None,
        base_url: str = "https://app.voxelai.com",
    ) -> List["OrganizationDailySummaryJob"]:
        """Factory function which returns list of jobs to be run.

        1. Invoke hourly via external scheduler (assume this is reliable)
        2. Get a list of organizations who have a summary email due this hour
        3. Group organizations into "jobs" based on summary type

        Args:
            invocation_timestamp (datetime): timestamp of when function was invoked.
            base_url (str): Base URL to use in content links.

        Returns:
            List of ready-to-run job instances.
        """

        if not invocation_timestamp:
            invocation_timestamp = timezone.now()

        logger.info("Getting jobs to run for organization daily summaries")

        # Get timezones of all organizations
        org_timezones = (
            Organization.objects.filter(is_sandbox=False)
            .values("id", "timezone")
            .distinct()
        )

        logger.info(f"org_timezones: {str(org_timezones)}")

        jobs = []
        for record in org_timezones:
            org_id = record.get("id")
            tz = record.get("timezone")
            if not tz:
                logger.error(
                    f"No timezone present for organization (ID: {org_id})"
                )
                continue

            local_hour = invocation_timestamp.astimezone(
                pytz.timezone(tz)
            ).hour

            logger.info(
                f"Organization ID: {org_id}, UTC hour: {invocation_timestamp.hour}, Local hour: {local_hour}"
            )

            if local_hour == cls.SEND_AT_LOCAL_HOUR:
                jobs.append(
                    OrganizationDailySummaryJob(
                        organization_id=record.get("id", -1),
                        invocation_timestamp=invocation_timestamp,
                        base_url=base_url,
                    )
                )
                logger.info(f"Queued job for organization #{org_id}")
            else:
                logger.info(f"No summaries due for organization #{org_id}")

        logger.info(f"Summary job count: {len(jobs)}")
        return jobs

    def run(self) -> None:
        """Runs the job.

        1. Get list of eligible users
        2. If zero, log and exit
        3. Generate user-specific data
        4. Send email
        5. Log activity

        TODO: add monitoring, retries, smarter batching, general robustness
        """

        logger.info(f"Running daily summary job for: {self.organization.name}")

        organization_summary_data_by_site = self.get_data()
        site_email_groups = self.get_site_email_groups()
        logs = self.get_logs()

        self.send_email_to_site_groupings(
            site_email_groups, organization_summary_data_by_site
        )

        NotificationLog.objects.bulk_create(logs)

    def get_logs(self) -> List[NotificationLog]:
        logs = []

        logger.info(f"Recipient count: {len(self.recipients)}")
        for recipient in self.recipients:
            logs.append(
                NotificationLog(
                    user=recipient,
                    sent_at=timezone.now(),
                    category=self.notification_category,
                )
            )

        return logs

    def get_site_email_groups(self) -> Dict[FrozenSet[Zone], List[str]]:
        site_email_groups = {
            frozenset(self.organization.sites): [
                settings.CUSTOMER_SUMMARY_EMAILS_BCC
            ],
        }

        for recipient in self.recipients:
            # get all sites associated with recipient
            recipient_sites = frozenset(
                recipient.zones.filter(
                    active=True,
                    zone_type=ZoneType.SITE,
                    organization__key=self.organization.key,
                )
            )

            if len(recipient_sites) == 0:
                logger.info(f"No associated sites for user: {recipient.email}")
                continue

            if not site_email_groups.get(recipient_sites):
                site_email_groups[recipient_sites] = []
            site_email_groups[recipient_sites].append(recipient.email)

        return site_email_groups

    def get_data(self) -> Dict[Zone, SummaryData]:
        summary_data_per_site = {}

        for site in self.organization.sites:
            logger.info(
                f"Localized invocation: {self.localized_invocation_timestamp.isoformat()}"
            )
            logger.info(
                f"Localized job start: {self.localized_start.isoformat()}"
            )
            logger.info(f"Localized job end: {self.localized_end.isoformat()}")
            site_series_data = get_series(
                organization=self.organization,
                zone=site,
                from_utc=self.localized_start,
                to_utc=self.localized_end,
                group_by="hour",
                current_user=None,
            )

            high_count = 0
            medium_count = 0
            low_count = 0

            def to_title(key: str):
                return " ".join(key.split("_")).title()

            incident_counts: Dict[str, Dict[str, Union[int, str]]] = {}

            # Initial value for incident counts
            for incident_type in self.organization.enabled_incident_types:
                key = incident_type.key
                incident_counts[key] = dict(
                    count=0,
                    url=self._filtered_incidents_url(
                        INCIDENT_TYPE=format_param_value([key]),
                        REDIRECT_SITE_KEY=site.key,
                    ),
                    title=to_title(incident_type.name),
                )

            points = [point for point in site_series_data if bool(point)]

            for point in points:
                high_count += point["priority_counts"]["high_priority_count"]
                medium_count += point["priority_counts"][
                    "medium_priority_count"
                ]
                low_count += point["priority_counts"]["low_priority_count"]

                incident_type_counts = point["incident_type_counts"]
                for key in incident_type_counts:
                    if key not in incident_counts:
                        incident_counts[key] = dict(
                            count=incident_type_counts[key],
                            url=self._filtered_incidents_url(
                                INCIDENT_TYPE=format_param_value([key]),
                                REDIRECT_SITE_KEY=site.key,
                            ),
                            title=to_title(key),
                        )
                    else:
                        incident_counts[key]["count"] += incident_type_counts[
                            key
                        ]

            # Convert dict to list for convenience rendering
            list_incident_counts = []
            for value in incident_counts.values():
                list_incident_counts.append(value)

            summary_data_per_site[site] = SummaryData(
                site_name=site.name,
                dashboard_url=self._dashboard_url(REDIRECT_SITE_KEY=site.key),
                high_count=high_count,
                high_count_url=self._filtered_incidents_url(
                    PRIORITY=format_param_value(["high"]),
                    REDIRECT_SITE_KEY=site.key,
                ),
                medium_count=medium_count,
                medium_count_url=self._filtered_incidents_url(
                    PRIORITY=format_param_value(["medium"]),
                    REDIRECT_SITE_KEY=site.key,
                ),
                low_count=low_count,
                low_count_url=self._filtered_incidents_url(
                    PRIORITY=format_param_value(["low"]),
                    REDIRECT_SITE_KEY=site.key,
                ),
                incident_counts=list_incident_counts,
                highlighted_string=self._get_highlighted_string(site),
                highlighted_thumbnail=self._get_highlighted_thumbnail(),
                highlighted_url=self._get_hightlighted_url(
                    EXTRAS=format_param_value(["HIGHLIGHTED"]),
                    REDIRECT_SITE_KEY=site.key,
                ),
            )

        return summary_data_per_site

    def _get_highlighted_thumbnail(self) -> str:
        """
        Returns a public static video thumbnail URL

        Returns:
            str: A thumbnail URL
        """
        # trunk-ignore(pylint/C0301)
        url = "https://voxel-public-assets.s3.us-west-2.amazonaws.com/incident_thumbnail_blurred_with_play_icon.jpg"
        return url

    def _get_highlighted_string(self, site) -> str:
        """Gets a string that is used in the email

        Args:
            site (Zone): Key of zone of incident

        Returns:
            str: A string that will be used in the email
        """
        start_timestamp = self.localized_start
        end_timestamp = self.localized_end
        highlighted_event_count = (
            site.incidents.from_timestamp(start_timestamp)
            .to_timestamp(end_timestamp)
            .filter(highlighted=True)
            .count()
        )
        if highlighted_event_count == 1:
            return f"{highlighted_event_count} new highlighted event"
        if highlighted_event_count > 1:
            return f"{highlighted_event_count} new highlighted events"
        return ""

    def _get_hightlighted_url(self, *_: None, **kwargs: Any) -> str:
        """Gets a URL

        Returns:
            str: A URL string to go incident page with HIGHLIGHTED in the filter parameter
        """
        query = urlencode({**kwargs})
        return f"{self.base_url}/incidents?{query}"

    def _filtered_incidents_url(self, *_: None, **kwargs: Any) -> str:
        time_range_params = {
            "startDate": format_param_value(
                self.localized_start.strftime("%Y-%m-%d")
            ),
            "endDate": format_param_value(
                self.localized_end.strftime("%Y-%m-%d")
            ),
        }
        query = urlencode({**kwargs, **time_range_params})
        return f"{self.base_url}/incidents?{query}"

    def _dashboard_url(self, *_: None, **kwargs: Any) -> str:
        query = urlencode({**kwargs})
        return f"{self.base_url}/dashboard?{query}"

    def send_email(self, to_emails: List[str], data: SummaryDataAggregate):
        SendGridClient().send_email_with_template(
            from_email=settings.DEFAULT_FROM_EMAIL,
            to_emails=to_emails,
            subject=self.notification_subject,
            template_id=DAILY_SUMMARY_TEMPLATE_ID,
            asm_group_id=DAILY_SUMMARY_ASM_GROUP_ID,
            **asdict(data),
        )
        logger.info(f"Organization daily summary email sent to {to_emails}")

    def send_email_to_site_groupings(
        self,
        site_groupings_to_email: Dict[FrozenSet[Zone], List[str]],
        summary_data_by_sites: Dict[Zone, SummaryData],
    ):
        for (sites, to_emails) in site_groupings_to_email.items():
            sites_data = [summary_data_by_sites[site] for site in sites]
            for email in to_emails:
                self.send_email(
                    to_emails=[email],
                    data=SummaryDataAggregate(
                        notification_subject=self.notification_subject,
                        organization_name=self.organization.name,
                        sites=sites_data,
                    ),
                )
