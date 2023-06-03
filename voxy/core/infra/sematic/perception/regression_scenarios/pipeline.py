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
from typing import Optional

import sematic

from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.infra.sematic.perception.graph_config import PerceptionPipelineConfig
from core.infra.sematic.perception.performance_evaluation import (
    EvaluationResults,
    evaluate_performance,
)
from core.infra.sematic.perception.run_inferences import execute_perception
from core.infra.sematic.shared.utils import PipelineSetup
from core.structs.scenario import ScenarioSet


@sematic.func
def pipeline(
    scenario_set: ScenarioSet,
    cache_key: str,
    run_uuid: str,
    perception_pipeline_config: PerceptionPipelineConfig,
    experiment_config_path: Optional[str],
    pipeline_setup: PipelineSetup,
    perception_runner_context: PerceptionRunnerContext,
) -> EvaluationResults:
    """# Map scenarios over the perception inference, evaluate, and summarize.

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
    - **experiment_config_path**:
        Path to YAML for the experiment config.

    ## Returns
    A summary of the model's performance, including precision/recall results.
    """
    aggregated_inferences = execute_perception(
        scenario_set,
        cache_key,
        run_uuid,
        perception_pipeline_config,
        experiment_config_path,
        perception_runner_context=perception_runner_context,
    )

    return evaluate_performance(
        aggregated_inferences,
        run_uuid=run_uuid,
        cache_key=cache_key,
        experiment_config_path=experiment_config_path,
    ).set(
        name="Evaluate Perception Performance",
        tags=[f"scenario-set:{scenario_set.name}"],
    )
