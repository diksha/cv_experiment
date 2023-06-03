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
import json
import unittest
from dataclasses import replace
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

from requests import Session
from requests.models import Response
from sematic.resolvers.silent_resolver import SilentResolver
from sematic.testing import mock_sematic_funcs
from sematic.utils.exceptions import ResolutionError

from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.incident_machine.utils import iter_machines
from core.infra.sematic.perception.performance_evaluation import (
    EvaluationResults,
)
from core.infra.sematic.perception.regression_scenarios.utils import (
    check_if_valid_incidents_in_portal,
    check_regression_set_complete,
    get_camera_uuid_model_map,
    run_and_evaluate_results,
    uncovered_incident_machines,
)
from core.infra.sematic.perception.run_inferences import (
    PerceptionPipelineConfig,
    execute_perception,
)
from core.infra.sematic.shared.utils import PipelineSetup
from core.scenarios.evaluate_performance import monitor_incident_map
from core.structs.incident import Incident
from core.structs.scenario import Scenario, ScenarioSet

EXAMPLE_CAMERA_UUID = "americold/modesto/0001/cha"


class UtilsTest(unittest.TestCase):
    @patch(
        "core.infra.sematic.perception.regression_scenarios.utils.load_yaml_with_jinja"
    )
    def test_get_camera_uuid_model_map(self, mock_yaml_load):
        # TEST PIGGYBACK AND BAD POSTURE
        mock_yaml_load.return_value = {
            "camera_uuid": "americold/modesto/0001/cha",
            "incident": {
                "state_machine_monitors_requested": [
                    "bad_posture",
                    "piggyback",
                ]
            },
        }
        cameras_to_incident = get_camera_uuid_model_map(
            ["americold/modesto/0001/cha"]
        )
        self.assertEqual(
            cameras_to_incident,
            (
                {
                    "americold/modesto/0001/cha": {
                        "carry_object",
                        "yolo",
                        "alpha_pose",
                        "door",
                    }
                },
                {"BAD_POSTURE", "PIGGYBACK"},
            ),
        )

        # TEST RANDOM SPILL IS UNSUPPORTED
        mock_yaml_load.return_value = {
            "camera_uuid": "americold/modesto/0001/cha",
            "incident": {
                "state_machine_monitors_requested": [
                    "random_spill",
                    "piggyback",
                ]
            },
        }
        cameras_to_incident = get_camera_uuid_model_map(
            ["americold/modesto/0001/cha"]
        )
        self.assertEqual(
            cameras_to_incident,
            (
                {
                    "americold/modesto/0001/cha": {
                        "yolo",
                        "door",
                    },
                },
                {"PIGGYBACK"},
            ),
        )

    @patch(
        "core.utils.perception_portal_graphql_session.get_secret_from_aws_secret_manager"
    )
    @patch.object(Session, "post")
    def test_check_if_valid_incidents_in_portal(
        self, mock_session: MagicMock, mock_get_secrets: MagicMock
    ):
        not_in_list = [("camera_uuid", "incident")]
        current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
        mock_get_secrets.return_value = (
            "{"
            '"auth_url": "auth_url",'
            '"client_id": "client_id",'
            '"client_secret": "secret",'
            '"audience": "audience",'
            '"host": "host",'
            f'"gen_time": "{current_time}",'
            '"access_token": "token",'
            "}"
        )
        response2 = Response()
        response2.status_code = 200
        mapping = {
            "node": {
                "camera_uuid": "camera_uuid",
                "incident": "incident",
            }
        }
        resp = {
            "data": {
                "integrations": {
                    "filteredRawIncidents": {"edges": json.dumps(mapping)}
                }
            }
        }
        # trunk-ignore(pylint/W0212)
        response2._content = json.dumps(resp).encode("ascii")
        response_ = []
        response_.append(response2)
        mock_session.side_effect = response_
        self.assertEqual(
            check_if_valid_incidents_in_portal(not_in_list),
            [("incident", "camera_uuid")],
        )

    @patch(
        "core.utils.perception_portal_graphql_session.get_secret_from_aws_secret_manager"
    )
    @patch.object(Session, "post")
    @patch(
        "core.infra.sematic.perception.regression_scenarios.utils.get_active_cameras"
    )
    @patch(
        "core.infra.sematic.perception.regression_scenarios.utils.slack_notification_for_metrics"
    )
    def test_check_regression_set_complete_true(
        self,
        slack_notification_for_metrics: MagicMock,
        active_cameras: MagicMock,
        mock_session: MagicMock,
        mock_get_secrets: MagicMock,
    ):
        current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
        mock_get_secrets.return_value = (
            "{"
            '"auth_url": "auth_url",'
            '"client_id": "client_id",'
            '"client_secret": "secret",'
            '"audience": "audience",'
            '"host": "host",'
            f'"gen_time": "{current_time}",'
            '"access_token": "token",'
            "}"
        )
        response2 = Response()
        response2.status_code = 200
        mapping = {
            "node": {
                "camera_uuid": "americold/modesto/0001/cha",
                "incident": "incident",
            }
        }
        resp = {
            "data": {
                "integrations": {
                    "filteredRawIncidents": {"edges": json.dumps(mapping)}
                }
            }
        }
        response2._content = json.dumps(  # trunk-ignore(pylint/W0212)
            resp
        ).encode("ascii")
        response_ = [response2 for _ in range(9)]
        mock_session.side_effect = response_
        active_cameras.return_value = ["americold/modesto/0001/cha"]
        scenarios = [
            Scenario(
                camera_uuid="americold/modesto/0001/cha",
                incidents=["BAD_POSTURE"],
                video_uuid="fake_uuid1",
                scenario_for_incidents=["BAD_POSTURE"],
            ),
            Scenario(
                camera_uuid="americold/modesto/0001/cha",
                incidents=["NO_STOP_AT_INTERSECTION"],
                video_uuid="fake_uuid2",
                scenario_for_incidents=["NO_STOP_AT_INTERSECTION"],
            ),
            Scenario(
                camera_uuid="americold/modesto/0001/cha",
                incidents=["PARKING_DURATION"],
                video_uuid="fake_uuid2",
                scenario_for_incidents=["PARKING_DURATION"],
            ),
            Scenario(
                camera_uuid="americold/modesto/0001/cha",
                incidents=["OVERREACHING"],
                video_uuid="fake_uuid2",
                scenario_for_incidents=["OVERREACHING"],
            ),
            Scenario(
                camera_uuid="americold/modesto/0001/cha",
                incidents=["NO_STOP_AT_DOOR_INTERSECTION"],
                video_uuid="fake_uuid2",
                scenario_for_incidents=["NO_STOP_AT_DOOR_INTERSECTION"],
            ),
            Scenario(
                camera_uuid="americold/modesto/0001/cha",
                incidents=["DOOR_VIOLATION"],
                video_uuid="fake_uuid2",
                scenario_for_incidents=["DOOR_VIOLATION"],
            ),
            Scenario(
                camera_uuid="americold/modesto/0001/cha",
                incidents=["OPEN_DOOR_DURATION"],
                video_uuid="fake_uuid2",
                scenario_for_incidents=["OPEN_DOOR_DURATION"],
            ),
            Scenario(
                camera_uuid="americold/modesto/0001/cha",
                incidents=["PIGGYBACK"],
                video_uuid="fake_uuid2",
                scenario_for_incidents=["PIGGYBACK"],
            ),
            Scenario(
                camera_uuid="americold/modesto/0001/cha",
                incidents=["SPILL"],
                video_uuid="fake_uuid2",
                scenario_for_incidents=["SPILL"],
            ),
        ]
        self.assertTrue(
            check_regression_set_complete(
                ScenarioSet(
                    name="test_set",
                    scenarios=scenarios,
                )
            )
        )

    @patch(
        "core.utils.perception_portal_graphql_session.get_secret_from_aws_secret_manager"
    )
    @patch.object(Session, "post")
    @patch(
        "core.infra.sematic.perception.regression_scenarios.utils.get_active_cameras"
    )
    @patch(
        "core.infra.sematic.perception.regression_scenarios.utils.slack_notification_for_metrics"
    )
    def test_check_regression_set_complete_false(
        self,
        slack_notification_for_metrics: MagicMock,
        active_cameras: MagicMock,
        mock_session: MagicMock,
        mock_get_secrets: MagicMock,
    ):
        current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
        mock_get_secrets.return_value = (
            "{"
            '"auth_url": "auth_url",'
            '"client_id": "client_id",'
            '"client_secret": "secret",'
            '"audience": "audience",'
            '"host": "host",'
            f'"gen_time": "{current_time}",'
            '"access_token": "token",'
            "}"
        )
        response2 = Response()
        response2.status_code = 200
        mapping = {
            "node": {
                "camera_uuid": "americold/modesto/0001/cha",
                "incident": "incident",
            }
        }
        resp = {
            "data": {
                "integrations": {
                    "filteredRawIncidents": {"edges": json.dumps(mapping)}
                }
            }
        }
        response2._content = json.dumps(  # trunk-ignore(pylint/W0212)
            resp
        ).encode("ascii")
        response_ = [response2 for i in range(9)]
        mock_session.side_effect = response_
        active_cameras.return_value = ["americold/modesto/0001/cha"]
        self.assertFalse(
            check_regression_set_complete(
                ScenarioSet(
                    name="test_set",
                    scenarios=[],
                )
            )
        )

    def test_run_and_evaluate_results(self):
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
            result: EvaluationResults = run_and_evaluate_results(
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

    def test_run_and_evaluate_results_failed(self):
        with mock_sematic_funcs([execute_perception]) as mock_funcs:
            mock_funcs[
                execute_perception
            ].mock.side_effect = mock_execute_perception_failing
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
            with self.assertRaises(ResolutionError):
                run_and_evaluate_results(
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

    def test_monitor_incident_map_is_complete(self):
        """Test that all incident types are in the monitor_incident_map"""
        all_machines = iter_machines("all")
        for machine in all_machines:
            print(machine.NAME)
            if machine.NAME not in uncovered_incident_machines:
                incident_name = monitor_incident_map.get(machine.NAME, None)
                self.assertIsNotNone(incident_name)


def mock_execute_perception(
    scenario_set: ScenarioSet, *_, **__
) -> List[Scenario]:
    """Execution perception results mocking the perception system.

    Args:
        scenario_set (ScenarioSet): sccenario set to store results in
        *_: unused
        **__: unused

    Returns:
        List: inferred incidents
    """
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


def mock_execute_perception_failing(
    scenario_set: ScenarioSet, *_, **__
) -> List[Scenario]:
    """Execution perception results mocking the perception system with no inferred results.

    Args:
        scenario_set (ScenarioSet): sccenario set to store results in
        *_: unused
        **__: unused
    Returns:
        List: inferred incidents
    """
    return scenario_set.scenarios
