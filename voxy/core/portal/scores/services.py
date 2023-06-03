import typing as t
from datetime import datetime

from django.contrib.postgres.aggregates import ArrayAgg
from django.db import connections
from loguru import logger
from psycopg2 import sql

from core.portal.compliance.jobs.utils import fetch_as_dict
from core.portal.scores.calculation_methods import CalculationMethodMapping
from core.portal.scores.models.score_definition import ScoreDefinition
from core.portal.scores.models.site_score_config import SiteScoreConfig
from core.portal.zones.models.zone import Zone


def _calculate_site_event_scores(
    site: Zone, score_definition_ids: list[int], date_time: datetime
) -> dict[str, int]:
    """Calculate all the given scores for the given datetime for the given site

    Args:
        site (Zone): The relevant site
        score_definition_ids (list[int]): The relevant scores
        date_time (datetime): The relevant datetime

    Returns:
        dict[str, int]: A mapping between the event score's name and the event score value
    """
    scores = {}
    for (
        score_definition_name,
        per_definition_id,
        score_band_id,
        calculation_method,
        camera_ids,
    ) in (
        SiteScoreConfig.objects.filter(
            site__exact=site,
            camera__isnull=False,
            score_definition__in=score_definition_ids,
        )
        .values("score_definition_id")
        .annotate(camera_ids=ArrayAgg("camera"))
        .values_list(
            "score_definition__name",
            "score_definition__perceived_event_rate_definition_id",
            "score_definition__score_band_id",
            "score_definition__calculation_method",
            "camera_ids",
        )
    ):
        score = CalculationMethodMapping[calculation_method](
            perceived_event_rate_definition_id=per_definition_id,
            score_band_id=score_band_id,
            camera_ids=camera_ids,
            date_time=date_time,
        )
        scores.update({score_definition_name: score})

    return scores


# NOTE: Should we make from_time, start_time UTC?
def get_site_event_scores(
    site: Zone,
    date_time: datetime,
) -> dict[str, int]:
    """Given a site, fetch all the site-specific event scores

    Args:
        site (Site): The relevant site
        date_time (datetime): The effective calculation date

    Returns:
        dict[str, int]: A mapping between the event score's name and the event score value
    """
    site_score_config_id_to_score_definition_id = {}
    for (
        site_score_config_id,
        score_definition_id,
    ) in SiteScoreConfig.objects.filter(
        site=site,
        camera__isnull=True,
        score_definition__calculation_method__exact=(
            ScoreDefinition.CalculationMethod.THIRTY_DAY_EVENT_SCORE
        ),
    ).values_list(
        "id", "score_definition"
    ):
        site_score_config_id_to_score_definition_id.update(
            {site_score_config_id: score_definition_id}
        )

    score_definition_name_to_score_value = _calculate_site_event_scores(
        site,
        site_score_config_id_to_score_definition_id.values(),
        date_time=date_time,
    )

    return score_definition_name_to_score_value


QUERY = """
with camera_to_incident_types_cte AS (
SELECT
site_score_config.camera_id, perceived_event_rate_definition.incident_type_id as "incident_type_id", score_definition.score_band_id as score_band_id, site_id, api_organization.id as organization_id, coalesce(zones.timezone, api_organization.timezone) as local_timezone, score_definition.name AS score_name
FROM site_score_config
INNER JOIN score_definition ON site_score_config.score_definition_id = score_definition.id
INNER JOIN perceived_event_rate_definition ON score_definition.perceived_event_rate_definition_id = perceived_event_rate_definition.id
INNER JOIN zones ON site_id = zones.id
INNER JOIN api_organization ON zones.organization_id = api_organization.id
WHERE zones.active is True AND camera_id is NOT NULL AND (organization_id = ANY(%(organizations)s) OR zones.id = ANY(%(sites)s))
),
relevant_hourly_vpers AS (
SELECT time_bucket_start_timestamp, score_band_id, score_name, site_id, organization_id, numerator_value, denominator_value
FROM perceived_event_rate_hourly
INNER JOIN perceived_event_rate_definition ON perceived_event_rate_hourly.definition_id = perceived_event_rate_definition.id
INNER JOIN camera_to_incident_types_cte ON perceived_event_rate_definition.incident_type_id = camera_to_incident_types_cte.incident_type_id AND perceived_event_rate_hourly.camera_id = camera_to_incident_types_cte.camera_id
-- TODO(itay): localize here!
WHERE time_bucket_start_timestamp >= (%(start_date)s) AND time_bucket_start_timestamp < (%(end_date)s)
),
size_rollup AS (
SELECT
score_band_id,
score_name,
{grouping_sets},
sum(numerator_value) as total_numerator_value,
sum(denominator_value) as total_denominator_value
FROM relevant_hourly_vpers
GROUP BY score_band_id, score_name, GROUPING SETS ({grouping_sets})
),
custom_vpers AS (
SELECT {grouping_sets}, score_band_id, score_name, total_numerator_value / total_denominator_value as vper_value
FROM size_rollup
),
custom_scores AS (
SELECT DISTINCT ON ({grouping_sets}, custom_vpers.score_band_id,  vper_value) {grouping_sets},
custom_vpers.score_band_id, custom_vpers.score_name, vper_value, coalesce(score_value, 0) as calculated_score
FROM custom_vpers
LEFT JOIN score_band_range ON custom_vpers.score_band_id = score_band_range.score_band_id AND lower_bound_inclusive >= vper_value
ORDER BY {grouping_sets}, custom_vpers.score_band_id, vper_value, lower_bound_inclusive ASC
)
select *
FROM custom_scores
"""


def calculate_all_organizational_event_scores(
    organization_ids: list[int],
    start_date: datetime,
    end_date: datetime,
) -> dict[t.Any, t.Any]:
    """Calculate and aggregate all events scores for each provided organization

    Args:
        organization_ids (list[int]): organizations
        start_date (datetime): inclusive
        end_date (datetime): exclusive

    Returns:
        dict[t.Any, t.Any]: results
    """
    return _calculate_all_event_scores(
        organization_ids=organization_ids,
        site_ids=[],
        start_date=start_date,
        end_date=end_date,
        grouping_set=[sql.Identifier("organization_id")],
    )


def calculate_all_site_event_scores(
    site_ids: list[int],
    start_date: datetime,
    end_date: datetime,
) -> dict[t.Any, t.Any]:
    """Calculate and aggregate all events scores for each provided site

    Args:
        site_ids (list[int]): sites
        start_date (datetime): inclusive
        end_date (datetime): exclusive

    Returns:
        dict[t.Any, t.Any]: results
    """
    return _calculate_all_event_scores(
        organization_ids=[],
        site_ids=site_ids,
        start_date=start_date,
        end_date=end_date,
        grouping_set=[sql.Identifier("site_id")],
    )


def _calculate_all_event_scores(
    organization_ids: list[int],
    site_ids: list[int],
    start_date: datetime,
    end_date: datetime,
    grouping_set: list[str],
) -> dict[t.Any, t.Any]:
    """Calculate and aggregate all events scores for each provided organization and site

    Args:
        organization_ids (list[int]): organizations
        site_ids (list[int]): sites
        start_date (datetime): inclusive
        end_date (datetime): exclusive
        grouping_set (list[str]): aggregation levels
    Returns:
        dict[t.Any, t.Any]: results
    """
    results = []
    try:
        query_identifiers = {
            "grouping_sets": sql.SQL("),(").join(
                [sql.SQL(",").join(grouping_set)]
            )
        }
        query_params = {
            "organizations": organization_ids,
            "sites": site_ids,
            "start_date": start_date,
            "end_date": end_date,
        }
        # trunk-ignore-all(pylint/C0301)
        with connections["default"].cursor() as cursor:
            # trunk-ignore(semgrep/python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query)
            cursor.execute(
                sql.SQL(QUERY).format(**query_identifiers), query_params
            )
            # TODO(itay): Make fetch_as_dict more efficent with named tuple
            results = fetch_as_dict(cursor, query_params)
    # trunk-ignore(pylint/W0718)
    except Exception as err:
        logger.error(f"Something went wrong {err}")

    return results
