import argparse
import copy
import json
import os
import random

import psycopg2
from loguru import logger
from tqdm import tqdm

_SELECT_QUERY = """
SELECT
    "api_incident"."data",
    "api_incident"."uuid",
    "api_organization"."key",
    "zones"."key" AS "zone_key"
FROM
    "api_incident"
    INNER JOIN "api_organization" ON "api_organization"."id" = "api_incident"."organization_id"
    INNER JOIN zones ON "zones"."id"="api_incident"."zone_id"
WHERE
    ("api_incident"."data" -> 'video_gcs_path') IS NOT NULL
    AND ("api_incident"."data" -> 'video_s3_path') IS NULL
ORDER BY timestamp ASC
LIMIT 1000
"""

_UPDATE_QUERY = """
UPDATE "api_incident" SET "data" = %s WHERE "uuid" = %s
"""


def transfer_data(uuid, data_type, org_key, zone_key, bucket):
    """Helper method to mv data from temporary s3 location to formatted s3 location.

    Args:
        uuid (str): incident uuid
        data_type (str): type of data
        org_key (str): organization key of incident
        zone_key (str): zone key of incident
        bucket (str): S3 bucket to write to

    Raises:
        ValueError: raises value error if unrecognized data type

    Returns:
        str: s3 path that file was uploaded to
    """
    if data_type == "original_video":
        # trunk-ignore(pylint/C0301): ok for long line
        destination_s3_path = f"s3://{bucket}/{org_key}/{zone_key}/incidents/{uuid}_original_video.mp4"
        source_s3_path = f"s3://{bucket}/incidents/{uuid}_original_video.mp4"
    elif data_type == "video":
        destination_s3_path = (
            f"s3://{bucket}/{org_key}/{zone_key}/incidents/{uuid}_video.mp4"
        )
        source_s3_path = f"s3://{bucket}/incidents/{uuid}_video.mp4"
    elif data_type == "json":
        # trunk-ignore(pylint/C0301): ok for long line
        destination_s3_path = f"s3://{bucket}/{org_key}/{zone_key}/incidents/{uuid}_annotations.json"
        source_s3_path = f"s3://{bucket}/incidents/{uuid}_annotations.json"
    elif data_type == "jpg":
        destination_s3_path = f"s3://{bucket}/{org_key}/{zone_key}/incidents/{uuid}_thumbnail.jpg"
        source_s3_path = f"s3://{bucket}/incidents/{uuid}_thumbnail.jpg"
    else:
        raise ValueError("unrecognized data type")

    # trunk-ignore(bandit/B605): ok for shell cmd
    ret = os.system(f"aws s3 mv {source_s3_path} {destination_s3_path}")
    if ret != 0:
        raise ValueError("system call failed")

    return destination_s3_path


def main(args):
    """Operation to transfer from temporary S3 location to formatted S3 location

    Args:
        args (Args): command line arguments
    """
    conn = psycopg2.connect(
        database=args.db_name,
        user=args.db_username,
        password=args.db_password,
        host=args.db_host,
        port=args.db_port,
    )
    curr = conn.cursor()
    bucket = args.s3_bucket

    try:
        while True:
            curr.execute(_SELECT_QUERY)

            rows = curr.fetchall()
            if len(rows) == 0:
                logger.info("Finished!")
                break

            for row in tqdm(rows):
                data = row[0]
                uuid = row[1]
                org_key = row[2].lower()
                zone_key = row[3].lower()

                # trunk-ignore(bandit/B311): log 20% of incidents
                if random.randint(0, 100) % 5 == 0:
                    logger.info(
                        f"Copying over incidents in AWS bucket for incident {uuid}..."
                    )

                new_data = copy.deepcopy(data)
                if data.get("original_video_gcs_path", None):
                    s3_path = transfer_data(
                        uuid,
                        "original_video",
                        org_key,
                        zone_key,
                        bucket,
                    )
                    new_data["original_video_s3_path"] = s3_path

                if data.get("video_gcs_path", None):
                    s3_path = transfer_data(
                        uuid,
                        "video",
                        org_key,
                        zone_key,
                        bucket,
                    )
                    new_data["video_s3_path"] = s3_path

                if data.get("annotations_gcs_path", None):
                    s3_path = transfer_data(
                        uuid,
                        "json",
                        org_key,
                        zone_key,
                        bucket,
                    )
                    new_data["annotations_s3_path"] = s3_path

                if data.get("video_thumbnail_gcs_path", None):
                    s3_path = transfer_data(
                        uuid,
                        "jpg",
                        org_key,
                        zone_key,
                        bucket,
                    )
                    new_data["video_thumbnail_s3_path"] = s3_path

                if new_data != data:
                    curr.execute(_UPDATE_QUERY, [json.dumps(new_data), uuid])

                conn.commit()
    finally:
        curr.close()
        conn.close()


def parse_args():
    """Argument parser.

    Returns:
        args: returns a struct holding the args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3_bucket", type=str, required=True)
    parser.add_argument("--db_host", type=str, required=True)
    parser.add_argument("--db_password", type=str, required=True)
    parser.add_argument("--db_port", type=str, required=True)
    parser.add_argument("--db_username", type=str, required=True)
    parser.add_argument("--db_name", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    script_arguments = parse_args()
    main(script_arguments)
