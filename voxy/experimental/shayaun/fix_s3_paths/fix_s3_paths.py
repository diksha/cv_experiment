import argparse
import copy
import json

import psycopg2
from loguru import logger

_SELECT_QUERY = """
SELECT "api_incident"."data", "api_incident"."uuid" WHERE (NOT "api_incident"."experimental" AND ("api_incident"."data" -> 'video_s3_path') IS NOT NULL)
"""


_UPDATE_QUERY = """
UPDATE "api_incident" SET "data" = %s WHERE "uuid" = %s
"""


def main(args):
    """Operation to fix S3 file paths
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
    try:
        curr.execute(_SELECT_QUERY)

        rows = curr.fetchall()
        if len(rows) == 0:
            logger.info("no rows")
            return

        for row in rows:
            data = row[0]
            uuid = row[1]
            logger.info(f"Auditing incident s3 path for incident {uuid}...")

            new_data = copy.deepcopy(data)
            bucket = f"voxel-portal-{args.environment}"

            original_video_s3_path = data.get("original_video_s3_path", None)
            if (
                original_video_s3_path
                and "s3://" not in original_video_s3_path
            ):
                new_data[
                    "original_video_s3_path"
                ] = f"s3://{bucket}/{original_video_s3_path}"

            video_s3_path = data.get("video_s3_path", None)
            if video_s3_path and "s3://" not in video_s3_path:
                new_data["video_s3_path"] = f"s3://{bucket}/{video_s3_path}"

            annotations_s3_path = data.get("annotations_s3_path", None)
            if annotations_s3_path and "s3://" not in annotations_s3_path:
                new_data[
                    "annotations_s3_path"
                ] = f"s3://{bucket}/{annotations_s3_path}"

            video_thumbnail_s3_path = data.get("video_thumbnail_s3_path", None)
            if (
                video_thumbnail_s3_path
                and "s3://" not in video_thumbnail_s3_path
            ):
                new_data[
                    "video_thumbnail_s3_path"
                ] = f"s3://{bucket}/{video_thumbnail_s3_path}"

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
    parser.add_argument("--db_host", type=str, required=True)
    parser.add_argument("--db_password", type=str, required=True)
    parser.add_argument("--db_port", type=str, required=True)
    parser.add_argument("--db_username", type=str, required=True)
    parser.add_argument("--db_name", type=str, required=True)
    parser.add_argument("--environment", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    script_arguments = parse_args()
    main(script_arguments)
