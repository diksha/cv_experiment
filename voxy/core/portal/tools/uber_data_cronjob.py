# trunk-ignore-all(pylint/C0413,flake8/E402)
import csv
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import List, Union

import boto3
import django
from aws_lambda_powertools.utilities import parameters
from botocore.credentials import Credentials
from django.db import connections
from django.db.models.query import QuerySet

# do our django setup
os.environ.setdefault("ENVIRONMENT", "development")
if os.environ["ENVIRONMENT"] != "development":
    for k, v in json.loads(parameters.get_secret("portal")).items():
        os.environ.setdefault(k, v)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.portal.voxel.settings")
django.setup()


from django.db.models import ExpressionWrapper, F, FloatField, Prefetch, Q
from django.db.models.fields.json import KeyTextTransform
from django.db.models.functions import Cast, Coalesce

from core.portal.api.models.incident import Incident
from core.portal.api.models.incident_feedback import IncidentFeedback

QUERY = """
SELECT
    api_incident.uuid as incident_uuid,
    api_incidenttype.key as incident_type_key,
    zones.anonymous_key as site_uuid,
    api_organization.anonymous_key as organization_uuid,
    (CASE
        WHEN api_incident.data->'original_end_frame_relative_ms' IS NOT NULL
        THEN (api_incident.data->>'original_end_frame_relative_ms')::float
        ELSE (api_incident.data->>'end_frame_relative_ms')::float
    END - (api_incident.data->>'start_frame_relative_ms')::float) AS video_length,
    api_incident.created_at as incident_created_at,
    api_incidentfeedback.feedback_value as gt_value,
    j.review->>'feedback_value' as feedback_value,
    j.review->>'timestamp' as feedback_created_at,
    j.review->>'elapsed_milliseconds_between_reviews' as elapsed_milliseconds_between_reviews,
    (SELECT email FROM auth_user WHERE id = (j.review->>'user_id')::integer) as email
FROM api_incident
LEFT JOIN api_incidenttype ON api_incident.incident_type_id = api_incidenttype.id
LEFT JOIN zones ON api_incident.zone_id = zones.id
LEFT JOIN api_organization ON api_incident.organization_id = api_organization.id
LEFT JOIN api_incidentfeedback ON api_incident.id = api_incidentfeedback.incident_id,
jsonb_array_elements(api_incident.data->'shadow_reviews') AS j(review)
WHERE (api_incident.data->>'shadow_reviewed')::boolean = TRUE
AND j.review->>'timestamp' IS NOT NULL
AND (j.review->>'timestamp')::timestamptz BETWEEN %s AND %s
"""


def assume_role(role_arn: str, session_name: str) -> Credentials:
    """
    Assume an IAM role and get temporary AWS security credentials.

    This function uses the AWS Security Token Service (STS) to assume an IAM role and obtain
    temporary security credentials for the session. These credentials consist of an access key ID,
    a secret access key, and a security token.

    Args:
        role_arn (str): The Amazon Resource Name (ARN) of the role to assume.
        session_name (str): An identifier for the assumed role session.

    Returns:
        botocore.credentials.Credentials: The temp credentials for the assumed role.
    """
    sts_client = boto3.client("sts")
    assumed_role_object = sts_client.assume_role(
        RoleArn=role_arn, RoleSessionName=session_name
    )

    return assumed_role_object["Credentials"]


video_length_expression = ExpressionWrapper(
    Cast(
        Coalesce(
            KeyTextTransform("original_end_frame_relative_ms", "data"),
            KeyTextTransform("end_frame_relative_ms", "data"),
        ),
        FloatField(),
    )
    - Cast(KeyTextTransform("start_frame_relative_ms", "data"), FloatField()),
    output_field=FloatField(),
)


def get_values_for_hour(start_time_input, end_time_input) -> QuerySet:
    """This function takes in a start time and end time, and returns
    a QuerySet containing incidents that occurred within that time range.

    Args:
        start_time_input (datetime): The start time of the range.
        end_time_input (datetime): The end time of the range.

    Returns:
        queryset (QuerySet): A Django QuerySet containing the incidents.
    """
    feedbacks = Prefetch(
        "feedback",
        queryset=IncidentFeedback.objects.filter(
            Q(user__email__endswith="@uber.com")
            | Q(user__email__endswith="@ext.uber.com")
        ).prefetch_related("user"),
    )

    return (
        Incident.objects.filter(
            feedback__created_at__range=[start_time_input, end_time_input]
        )
        .prefetch_related(feedbacks)
        .annotate(
            incident_uuid=F("uuid"),
            incident_type_key=F("incident_type__key"),
            site_uuid=F("zone__anonymous_key"),
            organization_uuid=F("organization__anonymous_key"),
            incident_created_at=F("created_at"),
            video_length=video_length_expression,
        )
    )


def get_shadow_review_values_for_hour(
    start_time_input, end_time_input
) -> list:
    """This function takes in a start time and end time, and returns
    a QuerySet containing shadow reviews on incidents that occurred within that time range.

    Args:
        start_time_input (datetime): The start time of the range.
        end_time_input (datetime): The end time of the range.

    Returns:
        list (list): A Django QuerySet containing the shadow reviewed incidents.
    """
    start_time_input_str = start_time_input.isoformat()
    end_time_input_str = end_time_input.isoformat()

    with connections["default"].cursor() as cursor:
        cursor.execute(QUERY, [start_time_input_str, end_time_input_str])
        column_names = [desc[0] for desc in cursor.description]
        results = [dict(zip(column_names, row)) for row in cursor.fetchall()]

    return results


def prepare_feedback_row(
    incident: Incident, feedback: IncidentFeedback, headers: List[str]
) -> List[Union[str, int, float]]:
    """
    Prepares a row for a feedback.

    Parameters:
        incident (Incident): Incident object.
        feedback (IncidentFeedback): Feedback object.
        headers (list): List of headers.

    Returns:
        row (list): A list of row values.
    """

    incident_served_timestamp = datetime.fromtimestamp(
        feedback.incident_served_timestamp_seconds, tz=timezone.utc
    ).isoformat()

    row = {
        "incident_uuid": incident.incident_uuid,
        "incident_type_key": incident.incident_type_key,
        "site_uuid": incident.site_uuid,
        "organization_uuid": incident.organization_uuid,
        "feedback_value": feedback.feedback_value,
        "feedback_comment": feedback.feedback_text,
        "feedback_created_at": feedback.created_at,
        "elapsed_milliseconds_between_reviews": feedback.elapsed_milliseconds_between_reviews,
        "video_length": incident.video_length,
        "incident_created_at": incident.incident_created_at,
        "incident_served_timestamp": incident_served_timestamp,
        "email": feedback.user.email,
    }
    return [row[key] for key in headers]


def shadow_review_results_to_csv(results: List[dict], file: object):
    """Convert shadow review query results to CSV

    Args:
        results (List[dict]): _description_
        file (object): _description_
    """
    headers = [
        "incident_uuid",
        "incident_type_key",
        "site_uuid",
        "organization_uuid",
        "feedback_value",
        "feedback_created_at",
        "elapsed_milliseconds_between_reviews",
        "video_length",
        "incident_created_at",
        "email",
        "gt_value",
    ]

    writer = csv.writer(file)
    writer.writerow(headers)
    for row in results:
        writer.writerow([row.get(key, "") for key in headers])


def queryset_to_csv(
    queryset_input: QuerySet,
    file: object,
):
    """
    This function takes in a QuerySet and a file, and writes
    the contents of the QuerySet to a CSV file.

    Parameters:
        queryset_input (QuerySet): The QuerySet to write to the CSV file.
        file (obj): The name of the CSV file.
    """
    headers = [
        "incident_uuid",
        "incident_type_key",
        "site_uuid",
        "organization_uuid",
        "feedback_value",
        "feedback_comment",
        "feedback_created_at",
        "elapsed_milliseconds_between_reviews",
        "video_length",
        "incident_created_at",
        "incident_served_timestamp",
        "email",
    ]

    writer = csv.writer(file)
    writer.writerow(headers)
    for incident in queryset_input:
        for feedback in incident.feedback.all():
            writer.writerow(prepare_feedback_row(incident, feedback, headers))


def get_s3_bucket():
    """Get s3 bucket path depending on environment variable

    Returns:
        str: return bucket name
    """
    return "voxeldata"


def process_hourly_data():
    """Process the hourly data"""
    end_time = datetime.now(tz=timezone.utc)
    start_time = end_time - timedelta(hours=1)

    date_path = end_time.strftime("%Y/%m/%d/%H")

    hourly_queryset = get_values_for_hour(start_time, end_time)
    shadow_review_values = get_shadow_review_values_for_hour(
        start_time, end_time
    )

    credentials = assume_role(
        "arn:aws:iam::203670452561:role/uber_voxeldata_bucket_access",
        "UberDataPipelineSession",
    )
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )

    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".csv") as temp_file:
        queryset_to_csv(hourly_queryset, temp_file)

        # Ensure all data is written to the temp file before we read it again
        temp_file.flush()

        s3_client.upload_file(
            temp_file.name,
            get_s3_bucket(),
            f"reviewer_performance/{date_path}/operators.csv",
            ExtraArgs={"ACL": "bucket-owner-full-control"},
        )

    with tempfile.NamedTemporaryFile(mode="w+t", suffix=".csv") as temp_file:
        shadow_review_results_to_csv(shadow_review_values, temp_file)

        # Ensure all data is written to the temp file before we read it again
        temp_file.flush()

        s3_client.upload_file(
            temp_file.name,
            get_s3_bucket(),
            f"reviewer_performance/{date_path}/trainees.csv",
            ExtraArgs={"ACL": "bucket-owner-full-control"},
        )


if __name__ == "__main__":
    process_hourly_data()
