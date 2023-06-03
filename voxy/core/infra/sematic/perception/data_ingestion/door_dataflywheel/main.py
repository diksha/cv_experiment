#
# Copyright 2020-2023 Voxel Labs, Inc.
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
from typing import Optional

from core.infra.sematic.perception.data_ingestion.door_dataflywheel.pipeline import (
    door_dataflywheel,
)
from core.infra.sematic.shared.utils import (
    PipelineSetup,
    SematicOptions,
    resolve_sematic_future,
)


def main(
    camera_uuid: str,
    start_date: Optional[str],
    max_incidents: int,
    sematic_options: SematicOptions,
    overwrite_config_file: Optional[str] = None,
) -> int:
    """Launch the pipeline

    Args:
        camera_uuid: See command line help
        start_date: See command line help
        max_incidents: See command line help
        overwrite_config_file (Optional[str]): config that will overwrite dataflywheel configs
        sematic_options (SematicOptions): Options for sematic resolvers

    Returns:
        The return code that should be used for the process

    Raises:
        NotImplementedError: if cloud execution is attempted (added later)
    """
    # trunk-ignore-all(pylint/E1101)
    future = door_dataflywheel(
        camera_uuid=camera_uuid,
        start_date=start_date,
        max_incidents=max_incidents,
        metaverse_environment=os.environ.get(
            "METAVERSE_ENVIRONMENT", "INTERNAL"
        ),
        overwrite_config_file=overwrite_config_file,
        pipeline_setup=PipelineSetup(),
    ).set(
        name=f"Door data flywheel{' (smoke test)' if overwrite_config_file is not None else ''}",
        tags=["P2", f"camera:{camera_uuid}", f"query_start:{start_date}"],
    )
    resolve_sematic_future(
        future,
        sematic_options,
        block_run=overwrite_config_file is not None,
    )
    return 0


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
        default=None,
        required=False,
        help=(
            "Date (UTC) in  YYYY-MM-DD format (for example: 2005-02-15). Defaults to none"
        ),
    )
    parser.add_argument(
        "-max",
        "--max-incidents",
        type=int,
        default=10,
        help="Max number of incidents to query from portal",
    )
    parser.add_argument(
        "--overwrite_config_file",
        type=str,
        help="Config file with changed values",
        default=None,
    )
    SematicOptions.add_to_parser(parser)
    args, _ = parser.parse_known_args()
    sys.exit(
        main(
            args.camera_uuid,
            args.start_date,
            args.max_incidents,
            SematicOptions.from_args(args),
            args.overwrite_config_file,
        )
    )
