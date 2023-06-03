from datetime import datetime, timedelta

import mock
import pytest
import pytz
from django.utils import timezone

from core.portal.api.models.notification_log import NotificationLog
from core.portal.api.models.organization import Organization
from core.portal.notifications.clients.sendgrid import SendGridClient
from core.portal.notifications.enums import NotificationCategory
from core.portal.notifications.jobs.organization_daily_summary import (
    OrganizationDailySummaryJob,
    SummaryData,
    SummaryDataAggregate,
)
from core.portal.testing.factories import (
    OrganizationFactory,
    UserFactory,
    ZoneFactory,
    ZoneUserFactory,
)
from core.portal.zones.enums import ZoneType


def get_test_summary_data() -> SummaryDataAggregate:

    return SummaryDataAggregate(
        notification_subject="test_subject",
        organization_name="test_organization",
        sites=[
            SummaryData(
                site_name="test_site",
                high_count=1,
                high_count_url="high_count_test.com",
                medium_count=2,
                medium_count_url="med_count_test.com",
                low_count=1,
                low_count_url="low_count_test.com",
                incident_counts=[],
                highlighted_string="highlighted string",
                highlighted_thumbnail="thumbnail.com",
                highlighted_url="highlighted.com",
            ),
        ],
    )


def seed_organization(
    timezone_str: str, user_count: int, zone_count: int = 0
) -> OrganizationFactory:
    zones = [ZoneFactory(zone_type=ZoneType.SITE) for _ in range(zone_count)]

    users = []
    for _ in range(user_count):
        u = UserFactory(
            profile__timezone=timezone_str,
            profile__data={"receive_daily_summary_emails": True},
        )
        users.append(u)

        for z in zones:
            ZoneUserFactory(
                user=u,
                zone=z,
            )

    return OrganizationFactory(
        timezone=timezone_str, is_sandbox=False, users=users, zones=zones
    )


def get_test_job(
    user_count: int = 3,
    zone_count: int = 0,
    invocation_timestamp: datetime = None,
) -> OrganizationDailySummaryJob:
    org = seed_organization("US/Eastern", user_count, zone_count)
    if not invocation_timestamp:
        invocation_timestamp = datetime(
            2021, 12, 6, 9, 0, 0, tzinfo=pytz.timezone("US/Eastern")
        )

    return OrganizationDailySummaryJob(
        organization_id=org.id,
        invocation_timestamp=invocation_timestamp,
        base_url="foo",
    )


@pytest.mark.django_db
def test_get_jobs_to_run_returns_jobs_in_correct_timezone() -> None:
    seed_organization("US/Eastern", 3)
    seed_organization("US/Eastern", 3)
    seed_organization("US/Central", 3)
    seed_organization("US/Central", 3)
    seed_organization("US/Pacific", 3)
    seed_organization("US/Pacific", 3)

    jobs = OrganizationDailySummaryJob.get_jobs_to_run(
        invocation_timestamp=datetime(
            2021, 12, 6, 9, 0, 0, tzinfo=pytz.timezone("US/Eastern")
        )
    )

    # Should only return US/Eastern jobs
    assert len(jobs) == 2
    for job in jobs:
        assert job.organization.timezone == "US/Eastern"
        assert len(job.recipients) == 3


@pytest.mark.django_db
def test_get_jobs_to_run_returns_jobs_with_correct_time_range() -> None:
    seed_organization("US/Eastern", 3)

    jobs = OrganizationDailySummaryJob.get_jobs_to_run(
        invocation_timestamp=datetime(
            2021, 12, 6, 9, 0, 0, tzinfo=pytz.timezone("US/Eastern")
        )
    )

    # Job data range should cover the day prior to invocation timestamp
    assert len(jobs) == 1
    assert jobs[0].localized_start == datetime(
        2021, 12, 5, 0, 0, 0, 0, tzinfo=pytz.timezone("US/Eastern")
    )
    assert jobs[0].localized_end == datetime(
        2021, 12, 5, 23, 59, 59, 999999, tzinfo=pytz.timezone("US/Eastern")
    )


@pytest.mark.django_db
@mock.patch.object(
    OrganizationDailySummaryJob,
    "send_email_to_site_groupings",
)
@mock.patch.object(
    OrganizationDailySummaryJob,
    "get_data",
)
def test_run_sends_emails_to_expected_recipients(
    mocked_get_data: mock.Mock,
    mocked_send_email: mock.Mock,
) -> None:
    get_test_job(zone_count=1).run()
    assert mocked_send_email.call_count == 1


@pytest.mark.django_db
@mock.patch.object(
    OrganizationDailySummaryJob,
    "send_email_to_site_groupings",
)
@mock.patch.object(
    OrganizationDailySummaryJob,
    "get_data",
)
def test_run_logs_notification_details(
    mocked_get_data: mock.Mock,
    mocked_send_email: mock.Mock,
) -> None:
    # One log entry should be created for each recipient
    get_test_job(user_count=3).run()
    assert NotificationLog.objects.count() == 3


@pytest.mark.django_db
@mock.patch.object(SendGridClient, "__init__", return_value=None)
@mock.patch.object(SendGridClient, "send_email_with_template")
@mock.patch(
    "core.portal.notifications.jobs.organization_daily_summary.get_series"
)
def test_run_send_email_with_template_params(
    mock_get_series,
    mock_send_email,
    mock_client,
) -> None:
    job = get_test_job()
    job.send_email(
        to_emails=["test@voxelai.com"],
        data=get_test_summary_data(),
    )
    assert mock_send_email.called_with(
        {
            "to_emails": ["test@voxelai.com"],
            "organization_name": "test_organization",
        }
    )


@pytest.mark.django_db
def test_run_getting_site_email_groups() -> None:
    user_count = 3
    zone_count = 2
    job = get_test_job(zone_count=zone_count, user_count=user_count)
    for (sites, users_to_email) in job.get_site_email_groups().items():
        assert len(sites) == zone_count
        # user_count + internal google group
        assert len(users_to_email) == user_count + 1


@pytest.mark.django_db
@mock.patch.object(SendGridClient, "__init__", return_value=None)
@mock.patch.object(SendGridClient, "send_email_with_template")
def test_run_skips_recipient_when_recent_notification_sent(
    mock_send_email_with_template,
    mock_client,
) -> None:
    job = get_test_job(user_count=1, invocation_timestamp=timezone.now())

    # Get test recipient
    recipient = Organization.objects.get(
        id=job.organization_id
    ).active_users.first()
    assert recipient is not None

    # Add recent notification log entry for this recipient
    NotificationLog.objects.create(
        user=recipient,
        category=NotificationCategory.ORGANIZATION_DAILY_SUMMARY,
        sent_at=timezone.now() - timedelta(hours=1),
    )

    assert len(job.recipients) == 0
