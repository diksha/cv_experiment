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
import os
import sys
from typing import List, Optional

from core.infra.sematic.perception.data_ingestion.ingest_scenarios.pipeline import (
    pipeline,
)
from core.infra.sematic.shared.utils import (
    PipelineSetup,
    SematicOptions,
    resolve_sematic_future,
)
from core.ml.common.utils import (
    add_camera_uuid_parser_arguments,
    get_camera_uuids_from_arguments,
)


# trunk-ignore(pylint/R0913)
def main(
    incident_type: str,
    camera_uuids: Optional[List[str]],
    start_date: Optional[str],
    end_date: Optional[str],
    max_incidents: int,
    environment: str,
    experimental_incidents_only: bool,
    is_test: bool,
    fps: float,
    sematic_options: SematicOptions,
) -> int:
    """Start off sematic pipeline

    Args:
        incident_type: See command line help
        camera_uuids: See command line help
        start_date: See command line help
        end_date: See command line help
        max_incidents: See command line help
        environment: See command line help
        experimental_incidents_only: See command line help
        is_test: See command line help
        fps: See command line help
        sematic_options: Options for sematic resolvers

    Returns:
        The return code that should be used for the process
    """
    # trunk-ignore(pylint/E1101)
    future = pipeline(
        incident_type=incident_type,
        camera_uuids=camera_uuids,
        start_date=start_date,
        end_date=end_date,
        max_incidents=max_incidents,
        environment=environment,
        experimental_incidents_only=experimental_incidents_only,
        is_test=is_test,
        fps=fps,
        metaverse_environment=os.environ.get(
            "METAVERSE_ENVIRONMENT", "INTERNAL"
        ),
        pipeline_setup=PipelineSetup(),
    ).set(name="Ingest scenarios", tags=["P3"])
    resolve_sematic_future(future, sematic_options)
    return 0


def parse_args() -> argparse.Namespace:
    """Parses arguments for the binary

    Returns:
        argparse.Namespace: Command line arguments passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--incident-type",
        type=str,
        required=True,
        help="Incident type to query portal for",
    )
    parser.add_argument(
        "-s",
        "--start-date",
        type=str,
        default=None,
        nargs="?",
        const=None,
        help="Format 2021-10-25",
    )
    parser.add_argument(
        "-e",
        "--end-date",
        type=str,
        default=None,
        nargs="?",
        const=None,
        help="Format 2021-10-25",
    )
    parser.add_argument(
        "-max",
        "--max-incidents",
        type=int,
        default=10,
        help=(
            "Get max-incident number of invalid and valid videos from portal"
        ),
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="production",
        help="Environment operation is being run in. Defaults to production.",
    )
    parser.add_argument(
        "--experimental-incidents-only",
        type=str,
        default="false",
        help="Only query for experimental incidents",
    )
    parser.add_argument(
        "--is-test",
        type=str,
        default="true",
        help=(
            "Defines if videos should be used for training or "
            "testing, for the most part, this pipeline should "
            "rely on is-test == true"
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=-1.0,
        help=(
            "FPS used for videos sent to labeling. FPS < 0 means video is "
            "not sent for labeling. FPS == 0.0 means use original video FPS"
        ),
    )
    SematicOptions.add_to_parser(parser)
    parser = add_camera_uuid_parser_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cameras = []
    if args.camera_uuids:
        cameras = get_camera_uuids_from_arguments(
            args.camera_uuids, args.organization, args.location
        )
    sys.exit(
        main(
            args.incident_type,
            cameras,
            args.start_date,
            args.end_date,
            args.max_incidents,
            args.environment,
            args.experimental_incidents_only.upper() == "TRUE",
            args.is_test.upper() == "TRUE",
            args.fps,
            SematicOptions.from_args(args),
        )
    )
