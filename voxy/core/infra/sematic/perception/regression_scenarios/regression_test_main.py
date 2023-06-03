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
import sys
import uuid
from typing import List

from loguru import logger

from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.infra.sematic.perception.graph_config import PerceptionPipelineConfig
from core.infra.sematic.perception.regression_scenarios.utils import (
    check_regression_set_complete,
    run_and_evaluate_results,
)
from core.infra.sematic.shared.utils import (
    PipelineSetup,
    SematicOptions,
    resolve_sematic_future,
)
from core.utils.struct_utils.scenario_utils import from_configuration_file


def main(
    scenario_set_path: str,
    inference_cluster_size: int,
    video_uuids: List[int],
    cache_key: str,
    experiment_config_path: str,
    sematic_options: SematicOptions,
) -> int:
    """Launch the pipeline

    Args:
        scenario_set_path: See command line help
        inference_cluster_size: See command line help
        video_uuids: See command line help
        cache_key: See command line help
        experiment_config_path: See command line help
        sematic_options: Options for Sematic resolvers

    Returns:
        The return code that should be used for the process
    """
    scenario_set = from_configuration_file(scenario_set_path)
    perception_config = PerceptionPipelineConfig(
        inference_cluster_size=inference_cluster_size,
        video_uuid_filter=video_uuids,
    )
    run_uuid = str(uuid.uuid4())
    logger.info(
        f"Launching inference on up to {len(scenario_set.scenarios)} scenarios"
    )
    if not check_regression_set_complete(scenario_set):
        logger.warning("Regression set is not complete")
    perception_runner_context = PerceptionRunnerContext()
    future = run_and_evaluate_results(
        scenario_set,
        cache_key,
        run_uuid,
        perception_config,
        experiment_config_path,
        pipeline_setup=PipelineSetup(),
        perception_runner_context=perception_runner_context,
    ).set(
        name=f"Regression Scenarios for '{scenario_set.name}'",
        tags=["P0", f"scenario-set:{scenario_set.name}"],
    )
    resolve_sematic_future(future, sematic_options, block_run=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--inference_cluster_size",
        type=int,
        required=False,
        help=(
            "Inferences will be executed on a Ray cluster. This parameter specifies "
            "the number of nodes in that cluster."
        ),
        default=5,
    )
    SematicOptions.add_to_parser(parser)
    args, _ = parser.parse_known_args()
    sys.exit(
        main(
            "data/scenario_sets/regression/regression_scenarios.yaml",
            args.inference_cluster_size,
            [],
            str(uuid.uuid4()),
            None,
            SematicOptions.from_args(args),
        )
    )
