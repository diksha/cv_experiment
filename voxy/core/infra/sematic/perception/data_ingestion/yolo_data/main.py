#
# Copyright 2022 Voxel Labs, Inc.
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
from typing import Dict, List, Optional

import mergedeep
from loguru import logger

from core.infra.sematic.perception.data_ingestion.yolo_data.pipeline import (
    LightlyIngestionUserInput,
    ingest_object_detection_data,
)
from core.infra.sematic.shared.utils import (
    PipelineSetup,
    SematicOptions,
    resolve_sematic_future,
)
from core.utils.yaml_jinja import load_yaml_with_jinja


# trunk-ignore(pylint/R0913)
def main(
    organization: str,
    location: str,
    ingestion_datetime: datetime,
    config: Dict[str, object],
    sematic_options: SematicOptions,
    test_size: float,
    dry_run: bool,
    lightly_num_samples: int,
    specific_camera_uuids: Optional[List[str]],
) -> int:
    """Launch the pipeline

    Args:
        organization: See command line help
        location: See command line help
        ingestion_datetime: See command line help
        test_size: See command line help
        config: the configuration to load for the runner
        sematic_options: Options for running sematic
        dry_run: waits on the future to resolve to exit
        specific_camera_uuids: camera_uuids if we want to collect data for particular cameras
        lightly_num_samples: For specific_camera_uuids, lightly num of samples camera subsets

    Returns:
        The return code that should be used for the process

    Raises:
        NotImplementedError: if cloud execution is attempted (added later)
    """
    camera_batch_map = LightlyIngestionUserInput(
        lightly_num_samples, specific_camera_uuids
    )
    # trunk-ignore(pylint/E1101)
    future = ingest_object_detection_data(
        organization=organization,
        location=location,
        ingestion_datetime=ingestion_datetime,
        metaverse_environment=os.environ["METAVERSE_ENVIRONMENT"],
        config=config,
        pipeline_setup=PipelineSetup(),
        test_size=test_size,
        camera_batch_map=camera_batch_map,
    ).set(
        name=(
            f"Yolo data ingestion for {organization}/{location}"
            f"{' (smoke test)' if dry_run else ''}"
        ),
        tags=[
            f"organization:{organization}",
            f"location:{location}",
            f"time:{ingestion_datetime.strftime('%Y-%m-%d-%H-%M')}",
            "P2",
        ],
    )
    result = resolve_sematic_future(future, sematic_options, block_run=dry_run)

    logger.info(f"results of future are {result}")
    return 0


def get_production_config() -> dict:
    """
    Loads and returns the pipeline configs (production)

    Returns:
        dict: regular production config
    """
    return load_yaml_with_jinja(
        "core/infra/sematic/perception/data_ingestion/yolo_data/configs/production.yaml"
    )


def get_smoketest_config() -> dict:
    """
    Loads and returns the pipeline configs (smoketest and regular)

    Returns:
        dict: smoke test config
    """
    return load_yaml_with_jinja(
        "core/infra/sematic/perception/data_ingestion/yolo_data/configs/smoke_test.yaml"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--organization",
        type=str,
        required=True,
        help=("Organization (for example: americold)"),
    )
    parser.add_argument(
        "--location",
        type=str,
        required=True,
        help=("Location of cameras (for example: ontario)"),
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        required=False,
        help=("Size of test split between 0.0 and 1.0"),
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
        "--specific_camera_uuids",
        type=str,
        nargs="*",
        required=False,
        help=(
            "Provide specific camera_uuids if we only want to collect data for particular cameras"
        ),
    )
    parser.add_argument(
        "--lightly_num_samples",
        type=int,
        default=-1,
        required=False,
        help=(
            "For specific_camera_uuids, provide lightly number of samples for subset of cameras"
        ),
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
            organization=args.organization,
            location=args.location,
            ingestion_datetime=datetime_to_ingest,
            config=config_values,
            test_size=args.test_size,
            sematic_options=SematicOptions.from_args(args),
            dry_run=args.dry_run,
            lightly_num_samples=args.lightly_num_samples,
            specific_camera_uuids=args.specific_camera_uuids
            if args.specific_camera_uuids
            else None,
        )
    )
