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
import itertools
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz
from loguru import logger

from core.execution.graphs.replay import ReplayGraph
from core.execution.utils.aggregate_metrics import aggregate_incident_metrics
from core.execution.utils.get_production_incidents import (
    get_production_incidents,
)
from core.execution.utils.graph_config_utils import (
    get_merged_graph_config_for_camera_uuid,
    get_updated_local_graph_config,
)
from core.utils.logger import configure_logger
from lib.utils.fetch_camera_frames import fetch_camera_frames
from protos.perception.structs.v1.frame_pb2 import (  # trunk-ignore(pylint/E0611)
    CameraFrame,
)


def parse_args() -> argparse.Namespace:
    """
    Parse input commandline args

    Returns:
        argparse.Namespace: the parsed input commanline arguments
    """
    parser = argparse.ArgumentParser(description="Replay Graph Runner")
    parser.add_argument("--log_key", type=str, default="")
    parser.add_argument("--experiment_config_path", type=str, default=None)
    parser.add_argument("--logging_level", type=str, default="debug")
    parser.add_argument("--camera_uuid", type=str, default=None, required=True)
    parser.add_argument("--compare", action=argparse.BooleanOptionalAction)
    time_format = "%Y-%m-%d-%H-%M"
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, time_format).replace(
            tzinfo=pytz.UTC
        ),
        required=True,
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, time_format).replace(
            tzinfo=pytz.UTC
        ),
        default=None,
    )
    return parser.parse_args()


def frame_differentiator(camera_frame: CameraFrame) -> str:
    """
    Generates information to differentiate frames

    Args:
        camera_frame (CameraFrame): the camera frame to process

    Returns:
        str: the item to group by to differentiate perception runs
    """
    return camera_frame.run_uuid


def camera_frame_generator(args: argparse.Namespace) -> CameraFrame:
    """
    Generates a stream of CameraFrame protobufs

    Args:
        args (argparse.Namespace): the input commanline args

    Returns:
        Iterator[CameraFrame]: the stream of camera frame protos
    """
    return fetch_camera_frames(
        args.camera_uuid, start=args.start, end=args.end
    )


def main():
    """
    Main runner for running the replay graph
    """
    args = parse_args()
    logger_level = args.logging_level.upper()
    configure_logger(level=logger_level, serialize=False)

    logger.info("Starting")

    all_incidents = []
    for run_uuid, frame_generator in itertools.groupby(
        camera_frame_generator(args), key=frame_differentiator
    ):
        graph_config = get_updated_local_graph_config(
            str(uuid.uuid4()),
            None,
            run_uuid,
            args.log_key,
            "configs/graphs/replay.yaml",
        )
        camera_config = get_merged_graph_config_for_camera_uuid(
            graph_config, args.experiment_config_path, args.camera_uuid
        )
        incidents = ReplayGraph(camera_config, frame_generator).execute()
        all_incidents.extend(incidents)
        portal_incidents = (
            get_production_incidents(args.camera_uuid, args.start, args.end)
            if args.compare
            else []
        )
        if not args.compare:
            logger.warning(
                "NOTE: portal counts will not be accurate to compare against as the "
                "--compare was not passed"
            )

        logger.info(
            aggregate_incident_metrics(
                replay_incidents=incidents,
                production_incidents=portal_incidents,
            )
        )
    # now we make a dataframe and compare them
    incidents_df = pd.DataFrame.from_dict(
        {
            "incidents": [incident.to_dict() for incident in all_incidents],
            "start_timestamp": [
                incident.start_frame_relative_ms for incident in all_incidents
            ],
            "end_timestamp": [
                incident.end_frame_relative_ms for incident in all_incidents
            ],
            "is_cooldown": [
                incident.cooldown_tag for incident in all_incidents
            ],
        }
    )
    csv_location = Path(
        os.path.join(
            tempfile.gettempdir(),
            "var",
            "voxel",
            f"{str(uuid.uuid4())}_replay_summary.csv",
        )
    )
    Path(csv_location.parent).mkdir(parents=True, exist_ok=True)
    incidents_df.to_csv(str(csv_location))
    logger.info(f"Wrote out incident summary to : {csv_location}")

    # TODO: assess requirements for comparing the results here with the ones in portal

    logger.info("Complete")


if __name__ == "__main__":
    main()
