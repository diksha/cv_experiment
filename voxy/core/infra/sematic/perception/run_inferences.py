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
from dataclasses import replace
from typing import List, Optional

import ray
import sematic
from sematic.ee.ray import RayCluster, SimpleRayCluster

from core.execution.graphs.develop import DevelopGraph
from core.execution.utils.graph_config_utils import (
    get_scenario_graph_configs_from_scenario_structs,
)
from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.infra.sematic.perception.graph_config import PerceptionPipelineConfig
from core.infra.sematic.shared.resources import RAY_NODE_GPU_4CPU_16GB
from core.structs.scenario import Scenario, ScenarioSet

# images using perception can be large. This can mean it takes
# a while to start up pods for Ray workers.
_CLUSTER_ACTIVATE_TIMEOUT_SECONDS = 60 * 60


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func(standalone=True)
def execute_perception(
    scenario_set: ScenarioSet,
    cache_key: str,
    run_uuid: str,
    perception_pipeline_config: PerceptionPipelineConfig,
    experiment_config_path: Optional[str],
    perception_runner_context: PerceptionRunnerContext,
) -> List[Scenario]:
    """# Map scenarios over the perception inference
    ## Parameters
    - **scenario_set**:
        The set of scenarios to run inference on, along with a name for the grouping
        of these scenarios.
    - **cache_key**:
        A unique key that will control usage of the results cache for output of the
        perception node. All results will be cached under the given cache key, so
        changing the key essentially creates a new, empty cache.
    - **run_uuid**:
        A unique way of identifying a new run; the results will be stored using this run_uuid.
        It is not cached however and a new run with same uuid will overwrite the results of
        previous run.
    - **perception_pipeline_config**:
        Configurations for how the perception portion of the pipeline should behave.
        See the documentation for
        core.infra.sematic.perception.graph_config.PerceptionPipelineConfig
        for details on parameters
    ## Returns
    A list of evaluated scenarios, with perception inferences populated.
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    if (
        perception_pipeline_config.video_uuid_filter is not None
        and len(perception_pipeline_config.video_uuid_filter) > 0
    ):
        scenarios = [
            s
            for s in scenario_set.scenarios
            if s.video_uuid in perception_pipeline_config.video_uuid_filter
        ]
    else:
        scenarios = scenario_set.scenarios

    with RayCluster(
        config=SimpleRayCluster(
            n_nodes=perception_pipeline_config.inference_cluster_size,
            node_config=RAY_NODE_GPU_4CPU_16GB,
        ),
        activation_timeout_seconds=_CLUSTER_ACTIVATE_TIMEOUT_SECONDS,
    ):
        inferences_ray_refs = [
            _run_inference_on_single_scenario.remote(
                scenario=scenario,
                cache_key=cache_key,
                run_uuid=run_uuid,
                perception_pipeline_config=perception_pipeline_config,
                experiment_config_path=experiment_config_path,
                perception_runner_context=perception_runner_context,
            )
            for scenario in scenarios
        ]
        inferences: List[Scenario] = ray.get(inferences_ray_refs)

    return inferences


@ray.remote(num_gpus=1, max_retries=3)
def _run_inference_on_single_scenario(
    scenario: Scenario,
    cache_key: str,
    run_uuid: str,
    perception_pipeline_config: PerceptionPipelineConfig,
    experiment_config_path: str,
    perception_runner_context: PerceptionRunnerContext,
) -> Scenario:
    """Perform inference on a single scenario, from Ray.

    Args:
        scenario:
            The scenario to run inference on
        cache_key:
            A unique key that will control usage of the results cache for output of the
            perception node. All results will be cached under the given cache key, so
            changing the key essentially creates a new, empty cache.
        run_uuid:
            A unique way of identifying a new run; the results will be stored using this run_uuid.
            It is not cached however and a new run with same uuid will overwrite the results of
            previous run.
        perception_pipeline_config:
            Configurations for how the perception portion of the pipeline should behave.
            See the documentation for
            core.infra.sematic.perception.graph_config.PerceptionPipelineConfig
            for details on parameters
        experiment_config_path: Path to the experimental configuration yaml for the scenario set run
    Returns:
        The scenario, with inference results populated
    Raises:
        ValueError: if no graph config is found
    """
    graph_configs = get_scenario_graph_configs_from_scenario_structs(
        cache_key,
        run_uuid,
        [scenario],
        perception_pipeline_config.video_uuid_filter,
        experiment_config_path=experiment_config_path,
    )
    if len(graph_configs) != 1:
        raise ValueError(
            "Graph config could not be properly retrieved for scenario batch"
        )
    graph_config = graph_configs[0]
    result = DevelopGraph(
        config=graph_config,
        perception_runner_context=perception_runner_context,
    ).execute()
    return replace(scenario, inferred_incidents=result)
