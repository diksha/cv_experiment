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
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import sematic

from core.execution.utils.graph_config_utils import (
    get_scenario_graph_configs_from_scenario_structs,
)
from core.scenarios.evaluate_performance import MonitorPerformanceEvaluator
from core.scenarios.evaluate_performance import (
    evaluate_performance as execute_performance_evaluation,
)
from core.structs.scenario import Scenario


@dataclass
class EvaluationResult:
    """A summary of inference performance, according to a given filtering criteria.

    Each instance of this class corresponds to evaluation results for a specific
    subset of inferences. For example, one instance of this class might summarize
    performance for door violations, while another might summarize performance
    for hard hats violations.

    Attributes
    ----------
    precision:
        The number of true incident positives divided by the number of total incident
        positives
    recall:
        The number of true incident positives divided by the number of ground-truth
        incidents
    true_positive_rate:
        What fraction of actual (ground truth) incidents are identified by perception?
    false_positive_rate:
        What fraction of scenarios without a ground truth incident are incorrectly
        identified by perception as having an incident?
    false_negative_rate:
        What fraction of scenarios with a ground truth incident fail to have that
        incident identified by perception?
    """

    precision: Optional[float]
    recall: Optional[float]
    true_positive_rate: Optional[float]
    false_positive_rate: Optional[float]
    false_negative_rate: Optional[float]


@dataclass
class EvaluationResults:
    """A summary of inference performance, across several subsets of inferences.

    Attributes
    ----------
    aggregate:
        Evaluation results across ALL incidents
    """

    results_by_incident_type: Dict[str, EvaluationResult]


# trunk-ignore-begin(pylint/W9011,pylint/W9006,pylint/W9015)
@sematic.func
def evaluate_performance(
    scenarios: List[Scenario],
    run_uuid: str,
    cache_key: str,
    experiment_config_path: Optional[str],
) -> EvaluationResults:
    """# Given completed inferences paired with ground truth, summarize the performance

    ## Parameters
    - **scenarios**:
        The set of scenarios that inferences were run on, along with ground-truth and
        calculated incidents for the scenarios.
    - **cache_key**:
        A unique key that will control usage of the results cache for output of the
        perception node. All results will be cached under the given cache key, so
        changing the key essentially creates a new, empty cache.
    - **run_uuid**:
        A unique way of identifying a new run; the results will be stored using this run_uuid.
        It is not cached however and a new run with same uuid will overwrite the results of
        previous run.

    ## Returns
    A summary of the model's performance, including precision/recall results.
    """
    # trunk-ignore-end(pylint/W9011,pylint/W9006,pylint/W9015)
    scenario_dicts = [asdict(s) for s in scenarios]
    graph_configs = get_scenario_graph_configs_from_scenario_structs(
        cache_key,
        run_uuid,
        scenarios,
        None,
        experiment_config_path=experiment_config_path,
    )
    if len(graph_configs) != len(scenario_dicts):
        raise ValueError(
            "Did not get the expected graph configs from scenarios"
        )

    for scenario_dict, graph_config in zip(scenario_dicts, graph_configs):
        scenario_dict["config"] = graph_config

    detected_incidents = [s.inferred_incidents for s in scenarios]

    performance_by_incident_type = execute_performance_evaluation(
        scenarios=scenario_dicts,
        detected_incidents=detected_incidents,
    )
    return EvaluationResults(
        results_by_incident_type={
            k: _as_evaluation_result(v)
            for k, v in performance_by_incident_type.items()
        }
    )


def _as_evaluation_result(
    performance_evaluator: MonitorPerformanceEvaluator,
) -> EvaluationResult:
    """Convert a MonitorPerformanceEvaluator into an EvaluationResult

    Args:
        performance_evaluator: an evaluator populated with the results of
            an evaluation

    Returns:
        A summary of the performance
    """
    return EvaluationResult(
        precision=performance_evaluator.get_precision(),
        recall=performance_evaluator.get_recall(),
        true_positive_rate=performance_evaluator.get_tp_rate(),
        false_positive_rate=performance_evaluator.get_fp_rate(),
        false_negative_rate=performance_evaluator.get_fn_rate(),
    )
