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
import unittest
from dataclasses import replace

from sematic.resolvers.silent_resolver import SilentResolver
from sematic.testing import mock_sematic_funcs

from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.infra.sematic.perception.performance_evaluation import (
    EvaluationResults,
)
from core.infra.sematic.perception.regression_scenarios.pipeline import (
    pipeline,
)
from core.infra.sematic.perception.run_inferences import (
    PerceptionPipelineConfig,
    execute_perception,
)
from core.infra.sematic.shared.utils import PipelineSetup
from core.structs.incident import Incident
from core.structs.scenario import Scenario, ScenarioSet

EXAMPLE_CAMERA_UUID = "americold/modesto/0001/cha"


# trunk-ignore-begin(pylint/C0116,pylint/W9012,pylint/W9011)
class PipelineTest(unittest.TestCase):
    def test_produces_evaluation_results(self):
        with mock_sematic_funcs([execute_perception]) as mock_funcs:
            mock_funcs[
                execute_perception
            ].mock.side_effect = mock_execute_perception
            fake_scenario_set = ScenarioSet(
                name="test",
                scenarios=[
                    Scenario(
                        camera_uuid=EXAMPLE_CAMERA_UUID,
                        incidents=["fake"],
                        video_uuid="fake_uuid",
                        scenario_for_incidents=["fake"],
                    )
                ],
            )
            perception_runner_context = PerceptionRunnerContext(
                "triton.server.url"
            )
            result: EvaluationResults = pipeline(
                scenario_set=fake_scenario_set,
                cache_key="doesntmatter",
                run_uuid="alsodoesntmatter",
                perception_pipeline_config=PerceptionPipelineConfig(
                    inference_cluster_size=5,
                    video_uuid_filter=None,
                ),
                experiment_config_path=None,
                pipeline_setup=PipelineSetup(),
                perception_runner_context=perception_runner_context,
            ).resolve(SilentResolver())
        self.assertIsInstance(result, EvaluationResults)
        some_evaluation_result = list(
            result.results_by_incident_type.values()
        )[0]
        self.assertLessEqual(0, some_evaluation_result.precision)
        self.assertGreaterEqual(some_evaluation_result.precision, 1.0)

        self.assertLessEqual(0, some_evaluation_result.recall)
        self.assertGreaterEqual(some_evaluation_result.recall, 1.0)


def mock_execute_perception(scenario_set: ScenarioSet, *_, **__):
    scenarios = scenario_set.scenarios
    return [
        replace(
            s,
            inferred_incidents=[
                (Incident(incident_type_id=i)) for i in s.incidents
            ],
        )
        for s in scenarios
    ]


# trunk-ignore-end(pylint/C0116,pylint/W9012,pylint/W9011)
