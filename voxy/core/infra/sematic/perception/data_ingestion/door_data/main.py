#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import argparse
import datetime as dt
import os
import sys
from datetime import datetime
from typing import Optional

import mergedeep

from core.infra.sematic.perception.data_ingestion.door_data.pipeline import (
    ingest_door_data,
)
from core.infra.sematic.shared.utils import (
    PipelineSetup,
    SematicOptions,
    resolve_sematic_future,
)
from core.utils.yaml_jinja import load_yaml_with_jinja


# Too many arguments: (this is a simple launch pipeline not an api)
# trunk-ignore-all(pylint/R0913)
def main(
    camera_uuid: str,
    ingestion_datetime: datetime,
    kinesis_upload_s3_path: Optional[str],
    rerun_from: Optional[str],
    max_videos: int,
    test_size: float,
    config: dict,
    sematic_options: SematicOptions,
    dry_run: bool,
) -> int:
    """Launch the pipeline

    Args:
        camera_uuid: See command line help
        ingestion_datetime: See command line help
        kinesis_upload_s3_path: See command line help
        rerun_from: See command line help
        max_videos: See command line help
        test_size: See command line help
        config: The config to run the pipeline (defines constants)
        sematic_options: options for Sematic resolvers
        dry_run: waits until future has finished
                         resolving to return

    Returns:
        The return code that should be used for the process

    Raises:
        NotImplementedError: if cloud execution is attempted (added later)
    """
    # trunk-ignore-all(pylint/E1101)
    future = ingest_door_data(
        camera_uuid=camera_uuid,
        ingestion_datetime=ingestion_datetime,
        metaverse_environment=os.environ.get(
            "METAVERSE_ENVIRONMENT", "INTERNAL"
        ),
        kinesis_upload_s3_path=kinesis_upload_s3_path,
        max_videos=max_videos,
        test_size=test_size,
        config=config,
        pipeline_setup=PipelineSetup(),
    ).set(
        name=f"Door data ingestion{' (smoke test)' if dry_run else ''}",
        tags=[
            "P1",
            f"camera:{camera_uuid}",
            f"time:{ingestion_datetime.strftime('%Y-%m-%d-%H-%M')}",
        ],
    )
    resolve_sematic_future(future, sematic_options, block_run=dry_run)
    return 0


def get_production_config() -> dict:
    """
    Loads and returns the pipeline configs (production)

    Returns:
        dict: regular production config
    """
    return load_yaml_with_jinja(
        "core/infra/sematic/perception/data_ingestion/door_data/configs/production.yaml"
    )


def get_smoketest_config() -> dict:
    """
    Loads and returns the pipeline configs (smoketest and regular)

    Returns:
        dict: smoke test config
    """
    return load_yaml_with_jinja(
        "core/infra/sematic/perception/data_ingestion/door_data/configs/smoke_test.yaml"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--camera-uuid",
        type=str,
        required=True,
        help=(
            "Camera uuid :camera: (for example: americold/ontario/0005/cha)"
        ),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="yesterday",
        required=False,
        help=(
            "Date (UTC) in  YYYY-MM-DD format (for example: 2005-02-15). Defaults to yesterday"
        ),
    )
    parser.add_argument(
        "--offset-time",
        type=float,
        default=0.0,
        required=False,
        help=(
            "The time of day in military time (UTC). For example, "
            "6:00AM would be '6.0'. Defaults to 12 AM i.e. 0.0"
        ),
    )
    parser.add_argument(
        "--kinesis-upload-s3-path",
        type=str,
        default=None,
        required=False,
        help=(
            "To use data that was *already pulled* from Kinesis, this can be "
            "specified to be an S3 path containing the data from Kinesis. If "
            "not set, data will be pulled from Kinesis."
        ),
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        required=False,
        help=("Size of test split between 0.0 and 1.0"),
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        required=False,
        help="Maximum number of videos.  0 is unlimited.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Whether to run the smoke test"
    )
    SematicOptions.add_to_parser(parser)
    args, _ = parser.parse_known_args()
    if not args.start_date or args.start_date == "yesterday":
        args.start_date = (datetime.now() - dt.timedelta(hours=24)).strftime(
            "%Y-%m-%d"
        )
    datetime_to_ingest = datetime.strptime(
        args.start_date, "%Y-%m-%d"
    ) + dt.timedelta(hours=args.offset_time)

    config_values = get_production_config()
    if args.dry_run:
        mergedeep.merge(config_values, get_smoketest_config())

    sys.exit(
        main(
            args.camera_uuid,
            datetime_to_ingest,
            args.kinesis_upload_s3_path,
            args.rerun_from,
            args.max_videos,
            args.test_size,
            config_values,
            SematicOptions.from_args(args),
            args.dry_run,
        )
    )
