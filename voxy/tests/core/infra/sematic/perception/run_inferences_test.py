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
from typing import List, Optional
from unittest.mock import MagicMock, patch

from sematic.resolvers.silent_resolver import SilentResolver

from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.infra.sematic.perception.graph_config import PerceptionPipelineConfig
from core.infra.sematic.perception.run_inferences import (
    _run_inference_on_single_scenario,
    execute_perception,
)
from core.structs.incident import Incident
from core.structs.scenario import Scenario, ScenarioSet

EXAMPLE_CAMERA_UUID = "americold/modesto/0001/cha"


# trunk-ignore-begin(pylint/C0115,pylint/C0116,pylint/W9011,pylint/W9012,pylint/C0103)
class RunInferencesTest(unittest.TestCase):
    @patch("core.infra.sematic.perception.run_inferences.ray")
    @patch(
        "core.infra.sematic.perception.run_inferences._run_inference_on_single_scenario"
    )
    def test_produces_inferences(self, mock_single_inference, mock_ray):
        mock_ray.get.side_effect = lambda arg: arg
        cache_key = "some_fake_key"
        run_uuid = "a_fake_run_uuid"
        perception_runner_context = PerceptionRunnerContext(
            "triton.server.url"
        )
        perception_config = PerceptionPipelineConfig(
            inference_cluster_size=5,
            video_uuid_filter=None,
        )
        mock_single_inference.remote = mock_run_single_inference
        scenarios = [
            Scenario(
                camera_uuid=EXAMPLE_CAMERA_UUID,
                incidents=["fake"],
                video_uuid="fake_uuid1",
                scenario_for_incidents=["fake"],
            ),
            Scenario(
                camera_uuid=EXAMPLE_CAMERA_UUID,
                incidents=["fake"],
                video_uuid="fake_uuid2",
                scenario_for_incidents=["fake"],
            ),
        ]
        scenario_set = ScenarioSet(
            name="test_set",
            scenarios=scenarios,
        )

        result: List[Scenario] = execute_perception(
            scenario_set,
            cache_key,
            run_uuid,
            perception_config,
            experiment_config_path=None,
            perception_runner_context=perception_runner_context,
        ).resolve(SilentResolver())
        self.assertTrue(all(isinstance(s, Scenario) for s in result))
        self.assertTrue(all(s.has_inference_results() for s in result))
        self.assertSetEqual(
            {s.video_uuid for s in result},
            {s.video_uuid for s in scenarios},
        )

    @patch("core.infra.sematic.perception.run_inferences.DevelopGraph")
    def test_run_inference_on_single_scenario(self, mock_develop_graph):
        mock_incident = MagicMock()
        mock_incident.incident_type_id = "inferred_incident"
        mock_develop_graph.return_value.execute.return_value = [mock_incident]
        scenario = make_n_scenarios(1)[0]
        perception_runner_context = PerceptionRunnerContext(
            "triton.server.url"
        )

        # trunk-ignore-begin(pylint/W0212)
        # OK to use _function for a test of underlying function without Ray.
        inferred_scenario = _run_inference_on_single_scenario._function(
            # trunk-ignore-end(pylint/W0212)
            scenario=scenario,
            cache_key="foo",
            run_uuid="bar",
            perception_pipeline_config=PerceptionPipelineConfig(
                inference_cluster_size=1,
            ),
            experiment_config_path=None,
            perception_runner_context=perception_runner_context,
        )

        self.assertListEqual(
            [
                incident.incident_type_id
                for incident in inferred_scenario.inferred_incidents
            ],
            ["inferred_incident"],
        )


def make_n_scenarios(n, populate_inferences=False):
    extra_kwargs = (
        {"inferred_incidents": ["fake"]} if populate_inferences else {}
    )
    return [
        Scenario(
            camera_uuid=EXAMPLE_CAMERA_UUID,
            incidents=["fake"],
            video_uuid=str(i),
            scenario_for_incidents=["fake"],
            **extra_kwargs,
        )
        for i in range(n)
    ]


# trunk-ignore-begin(pylint/W0613)
def mock_run_single_inference(
    scenario: Scenario,
    cache_key: str,
    run_uuid: str,
    perception_pipeline_config: PerceptionPipelineConfig,
    experiment_config_path: Optional[str],
    perception_runner_context: PerceptionRunnerContext,
):
    return replace(
        scenario,
        inferred_incidents=[
            (Incident(incident_type_id=i)) for i in scenario.incidents
        ],
    )


# trunk-ignore-end(pylint/W0613)
# trunk-ignore-end(pylint/C0115,pylint/C0116,pylint/W9011,pylint/W9012,pylint/C0103)
