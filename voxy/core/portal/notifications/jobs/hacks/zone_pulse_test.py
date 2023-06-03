from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import pytest
from django.conf import settings
from django.utils import timezone

from core.portal.incidents.models.review_level import ReviewLevel
from core.portal.notifications.jobs.hacks.zone_pulse import (
    IncidentTypeCount,
    ZonePulseJob,
)
from core.portal.testing.factories import (
    CameraFactory,
    IncidentFactory,
    IncidentTypeFactory,
    OrganizationFactory,
    ZoneFactory,
)

mon = datetime(2022, 7, 18)
wed = datetime(2022, 7, 20)
sat = datetime(2022, 7, 23)
sun = datetime(2022, 7, 24)


@dataclass
class ScheduleConfig:
    invocation_hour: int
    invocation_minute: int
    day_count_tuples: List[Tuple[datetime, int]]
    start_hour_minute_second: Tuple[int, int, int] = None
    end_hour_minute_second: Tuple[int, int, int] = None


@pytest.mark.django_db
@pytest.mark.parametrize(
    "invocation_hour,invocation_minute,start_hour_minute_second,end_hour_minute_second,day_count_tuples",
    [
        (0, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (0, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (1, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (1, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (2, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (
            2,
            30,
            (23, 0, 0),
            (1, 59, 59),
            [(mon, 0), (wed, 1), (sat, 1), (sun, 0)],
        ),
        (3, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (3, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (4, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (4, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (5, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (5, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (6, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (6, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (7, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (
            7,
            30,
            (5, 0, 0),
            (6, 59, 59),
            [(mon, 1), (wed, 1), (sat, 1), (sun, 0)],
        ),
        (8, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (8, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (9, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (
            9,
            30,
            (7, 0, 0),
            (8, 59, 59),
            [(mon, 1), (wed, 1), (sat, 1), (sun, 0)],
        ),
        (10, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (10, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (11, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (
            11,
            30,
            (9, 0, 0),
            (10, 59, 59),
            [(mon, 1), (wed, 1), (sat, 1), (sun, 0)],
        ),
        (12, 0, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (12, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (13, 0, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (
            13,
            30,
            (11, 0, 0),
            (12, 59, 59),
            [(mon, 1), (wed, 1), (sat, 1), (sun, 0)],
        ),
        (14, 0, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (14, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (15, 0, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (
            15,
            30,
            (13, 0, 0),
            (14, 59, 59),
            [(mon, 1), (wed, 1), (sat, 1), (sun, 0)],
        ),
        (16, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (16, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (17, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (
            17,
            30,
            (15, 0, 0),
            (16, 59, 59),
            [(mon, 1), (wed, 1), (sat, 0), (sun, 0)],
        ),
        (18, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (18, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (19, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (
            19,
            30,
            (17, 0, 0),
            (18, 59, 59),
            [(mon, 1), (wed, 1), (sat, 0), (sun, 0)],
        ),
        (20, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (20, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (21, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (
            21,
            30,
            (19, 0, 0),
            (20, 59, 59),
            [(mon, 1), (wed, 1), (sat, 0), (sun, 0)],
        ),
        (22, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (22, 30, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (23, 00, None, None, [(mon, 0), (wed, 0), (sat, 0), (sun, 0)]),
        (
            23,
            30,
            (21, 0, 0),
            (22, 59, 59),
            [(mon, 1), (wed, 1), (sat, 0), (sun, 0)],
        ),
    ],
)
def test_get_jobs_to_run_returns_expected_jobs_at_laredo(
    invocation_hour,
    invocation_minute,
    start_hour_minute_second,
    end_hour_minute_second,
    day_count_tuples,
) -> None:
    zone = ZoneFactory(key="LAREDO", name="Laredo", timezone="US/Central")

    for (invocation_day, expected_job_count) in day_count_tuples:
        invocation_timestamp = invocation_day.replace(
            hour=invocation_hour, minute=invocation_minute, tzinfo=zone.tzinfo
        )
        jobs = ZonePulseJob.get_jobs_to_run(invocation_timestamp)
        assert len(jobs) == expected_job_count
        for job in jobs:
            assert (
                job.localized_start_timestamp.hour
                == start_hour_minute_second[0]
            )
            assert (
                job.localized_start_timestamp.minute
                == start_hour_minute_second[1]
            )
            assert (
                job.localized_start_timestamp.second
                == start_hour_minute_second[2]
            )
            assert (
                job.localized_end_timestamp.hour == end_hour_minute_second[0]
            )
            assert (
                job.localized_end_timestamp.minute == end_hour_minute_second[1]
            )
            assert (
                job.localized_end_timestamp.second == end_hour_minute_second[2]
            )


@pytest.mark.django_db
def test_run_returns_expected_data() -> None:
    incident_type_1 = IncidentTypeFactory(key="foo", name="Foo")
    incident_type_2 = IncidentTypeFactory(key="bar", name="Bar")
    incident_type_3 = IncidentTypeFactory(key="bazz", name="Bazz")
    camera = CameraFactory()
    # trunk-ignore(pylint/E1101): does have member
    camera.incident_types.add(incident_type_1)
    # trunk-ignore(pylint/E1101): does have member
    camera.incident_types.add(incident_type_2)
    # trunk-ignore(pylint/E1101): does have member
    camera.incident_types.add(incident_type_3)

    org = OrganizationFactory(cameras=[camera])
    zone = ZoneFactory(
        key="LAREDO", name="Laredo", timezone="US/Central", organization=org
    )
    org.incident_types.add(incident_type_1)
    org.incident_types.add(incident_type_2)
    org.incident_types.add(incident_type_3)

    IncidentFactory(
        zone=zone,
        incident_type=incident_type_1,
        visible_to_customers=True,
        review_level=ReviewLevel.GREEN,
    )
    IncidentFactory(
        zone=zone,
        incident_type=incident_type_2,
        visible_to_customers=True,
        review_level=ReviewLevel.GREEN,
    )
    IncidentFactory(
        zone=zone,
        incident_type=incident_type_3,
        visible_to_customers=True,
        review_level=ReviewLevel.GREEN,
    )

    job = ZonePulseJob(
        zone_id=zone.id,
        timespan_desc="1pm - 3pm",
        invocation_timestamp=timezone.now(),
        localized_start_timestamp=timezone.now() - timedelta(hours=1),
        localized_end_timestamp=timezone.now(),
        base_url=settings.BASE_URL,
    )
    data = job.get_data()

    assert data.timespan_desc == "1pm - 3pm"
    assert data.dashboard_url == settings.BASE_URL

    expected_counts = [
        IncidentTypeCount("bar", "Bar", 1),
        IncidentTypeCount("bazz", "Bazz", 1),
        IncidentTypeCount("foo", "Foo", 1),
    ]

    assert expected_counts == sorted(
        data.incident_type_counts, key=lambda x: x.key
    )
