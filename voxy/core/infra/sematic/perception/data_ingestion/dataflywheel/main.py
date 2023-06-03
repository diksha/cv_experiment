#
# Copyright 2023 Voxel Labs, Inc.
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

from core.infra.sematic.perception.data_ingestion.dataflywheel.pipeline import (
    run_dataflywheel,
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
from core.structs.model import ModelCategory
from core.structs.task import TaskPurpose


def parse_args() -> argparse.Namespace:
    """Parses arguments for the binary

    Returns:
        argparse.Namespace: Command line arguments passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-purpose",
        type=str,
        help="The task purpose to use for the dataflywheel task",
        choices=TaskPurpose.names(),
        required=True,
    )
    parser.add_argument(
        "--overwrite_config_file",
        type=str,
        help="Config file with changed values",
        default=None,
    )
    parser.add_argument(
        "--model-category",
        type=str,
        help="The model category ",
        choices=ModelCategory.names(),
        required=True,
    )
    parser.add_argument(
        "--notify",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to notify slack",
    )
    parser.add_argument(
        "-s",
        "--start_date",
        type=str,
        default=None,
        help="Format 2021-10-25",
    )
    parser.add_argument(
        "-e",
        "--end_date",
        type=str,
        default=None,
        help="Format 2021-10-25",
    )
    parser.add_argument(
        "-max",
        "--max_incidents",
        type=int,
        default=10,
        help="Max number of incidents",
    )
    SematicOptions.add_to_parser(parser)
    parser = add_camera_uuid_parser_arguments(parser)
    return parser.parse_args()


# trunk-ignore(pylint/R0913)
def main(
    task_purpose: str,
    model_category: str,
    camera_uuids: List[str],
    should_notify: bool,
    start_date: Optional[str],
    end_date: Optional[str],
    max_incidents: int,
    sematic_options: SematicOptions,
    overwrite_config_file: Optional[str] = None,
):
    """
    Generates a dataflywheel object and executes it
    Args:
        task_purpose (str): task purpose to use for dataflywheel
        model_category (str): model category
        camera_uuids (List[str]): list of camera uuids to initialize the task
        should_notify (bool): notify slack flag
        start_date (Optional[str]): start date to query incidents from portal
        end_date (Optional[str]): end date to query incidents from portal
        max_incidents (int): max number of incidents to query portal for
        overwrite_config_file (Optional[str]): config that will overwrite dataflywheel configs
        sematic_options (SematicOptions): sematic options
    """
    # Sematic updates these with futures
    # trunk-ignore-all(pylint/E1101)
    future = run_dataflywheel(
        task_purpose=task_purpose,
        model_category=model_category,
        camera_uuids=camera_uuids,
        should_notify=should_notify,
        start_date=start_date,
        end_date=end_date,
        max_incidents=max_incidents,
        metaverse_environment=os.environ.get(
            "METAVERSE_ENVIRONMENT", "INTERNAL"
        ),
        overwrite_config_file=overwrite_config_file,
        pipeline_setup=PipelineSetup(),
    ).set(
        name=(
            f"Data Flywheel for {task_purpose}"
            f"{' (smoke test)' if overwrite_config_file is not None else ''}"
        ),
        tags=["P1", f"cameras:{camera_uuids}", f"start_date:{start_date}"],
    )
    resolve_sematic_future(
        future,
        sematic_options,
        block_run=overwrite_config_file is not None,
    )


if __name__ == "__main__":
    args = parse_args()
    cameras = get_camera_uuids_from_arguments(
        args.camera_uuids, args.organization, args.location
    )
    sys.exit(
        main(
            args.task_purpose,
            args.model_category,
            cameras,
            args.notify,
            args.start_date,
            args.end_date,
            args.max_incidents,
            sematic_options=SematicOptions.from_args(args),
            overwrite_config_file=args.overwrite_config_file,
        )
    )
