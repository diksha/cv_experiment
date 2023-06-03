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
import uuid

from loguru import logger
from tqdm import tqdm

from core.execution.graphs.develop import DevelopGraph
from core.execution.utils.graph_config_utils import (
    get_scenario_graph_configs_from_file,
    get_updated_local_graph_config,
)
from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.scenarios.evaluate_performance import (
    compare_performance,
    evaluate_performance,
    log_performance_results,
)
from core.utils.logger import configure_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Develop Graph Runner")
    parser.add_argument("--cache_key", type=str, default=str(uuid.uuid4()))
    parser.add_argument("--log_key", type=str, default="")
    parser.add_argument("--scenarios_config_path", type=str, required=True)
    parser.add_argument("--experiment_config_path", type=str, default=None)
    parser.add_argument(
        "--video_uuids", metavar="N", type=str, nargs="+", default=[]
    )
    parser.add_argument("--generate_scenarios", action="store_true")
    parser.add_argument("--max_concurrency", type=int, default=4)
    parser.add_argument("--logging_level", type=str, default="debug")
    parser.add_argument("--portal", dest="portal", action="store_true")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true")
    parser.add_argument(
        "--video_writer", dest="enable_video_writer", action="store_true"
    )
    parser.add_argument("--compare_config_path", type=str, default=None)
    parser.add_argument(
        "--compare_cache_key", type=str, default=str(uuid.uuid4())
    )
    parser.add_argument("--triton_server_url", type=str, default="")
    return parser.parse_args()


def _execute_graph(graph_configs, max_concurrency, perception_runner_context):
    results = []
    for config in tqdm(graph_configs):
        results.append(
            DevelopGraph(
                config=config,
                perception_runner_context=perception_runner_context,
            ).execute()
        )
    return results


def main():
    args = parse_args()
    logger_level = args.logging_level.upper()
    configure_logger(level=logger_level, serialize=False)
    perception_runner_context = (
        PerceptionRunnerContext(_triton_server_url=args.triton_server_url)
        if args.triton_server_url
        else PerceptionRunnerContext()
    )

    logger.info("Starting")

    run_uuid = str(uuid.uuid4())

    # Update graph & scenario config
    graph_config = get_updated_local_graph_config(
        args.cache_key, args.portal, run_uuid, args.log_key
    )
    (
        scenario_graph_configs,
        config_all_scenarios,
    ) = get_scenario_graph_configs_from_file(
        graph_config,
        args.experiment_config_path,
        args.scenarios_config_path,
        args.video_uuids,
        args.enable_video_writer,
    )
    results = _execute_graph(
        scenario_graph_configs, args.max_concurrency, perception_runner_context
    )

    # If we want to run a comparison
    if args.compare_config_path:
        graph_config = get_updated_local_graph_config(
            args.compare_cache_key,
            args.portal,
            run_uuid,
            args.log_key,
        )
        (
            scenario_graph_configs,
            compare_all_scenarios,
        ) = get_scenario_graph_configs_from_file(
            graph_config,
            args.compare_config_path,
            args.scenarios_config_path,
            args.video_uuids,
        )

        compare_results = _execute_graph(
            scenario_graph_configs,
            args.max_concurrency,
            perception_runner_context,
        )

    # Evaluate performance
    if args.evaluate:
        performance_by_incident = evaluate_performance(
            config_all_scenarios,
            results,
        )
        log_performance_results(performance_by_incident)

        # Evaluate performance for comparison config
        if args.compare_config_path:
            compare_performance_by_incident = evaluate_performance(
                compare_all_scenarios,
                compare_results,
            )
            log_performance_results(compare_performance_by_incident)

            # Compare performance
            compare_performance(
                performance_by_incident, compare_performance_by_incident
            )

    logger.info("Complete")


if __name__ == "__main__":
    main()
