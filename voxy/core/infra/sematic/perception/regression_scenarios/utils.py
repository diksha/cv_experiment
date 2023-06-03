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
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import sematic
from loguru import logger

from core.common.queries import ACTIVE_CAMERAS, FILTERED_RAW_INCIDENTS_QUERY
from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.infra.sematic.perception.graph_config import PerceptionPipelineConfig
from core.infra.sematic.perception.performance_evaluation import (
    EvaluationResults,
)
from core.infra.sematic.perception.regression_scenarios.pipeline import (
    pipeline,
)
from core.infra.sematic.shared.utils import PipelineSetup
from core.scenarios.evaluate_performance import monitor_incident_map
from core.structs.scenario import ScenarioSet
from core.utils.logging.slack.get_slack_webhooks import (
    get_perception_verbose_sync_webhook,
)
from core.utils.logging.slack.synchronous_webhook_wrapper import (
    SynchronousWebhookWrapper,
)
from core.utils.perception_portal_graphql_session import (
    PerceptionPortalSession,
)
from core.utils.yaml_jinja import load_yaml_with_jinja

# For models not getting affected by multiple incidents
# there is an automatic check and map for that would be
# example spill : "spill"
incident_to_model_type = {
    "piggyback": ["yolo", "door"],
    "safety_vest": ["yolo", "safety_vest"],
    "hard_hat": ["yolo", "hard_hat"],
    "intersection": ["yolo"],
    "no_ped_zone": ["yolo"],
    "aisle": ["yolo"],
    "door_intersection": ["door"],
    "door_violation": ["door"],
    "open_door": ["door"],
    "bad_posture": ["yolo", "alpha_pose", "carry_object"],
    "spill": ["spill"],
    "obstruction": ["obstruction"],
}

# We do not care about coverage for some incident machines
# for example: "random_spill" since it will take 60 mins to
# generate an incident
uncovered_incident_machines = {
    "random_spill",
}


@dataclass
class CoverageResults:
    """
    Coverage of regression scenario
    """

    incidents_coverage: float
    camera_model_coverage: float
    per_camera_coverage: dict
    per_model_coverage: dict


@sematic.func
def evaluate_results(
    evaluation_results: EvaluationResults,
) -> EvaluationResults:
    """Evaluate results from scenarios

    Args:
        evaluation_results (EvaluationResults): results of scenarios

    Raises:
        RuntimeError: if precision or recall is not 1

    Returns:
        EvaluationResults: results of evaluation
    """
    for (
        _,
        results,
    ) in evaluation_results.results_by_incident_type.items():
        if results.precision != 1 or results.recall != 1:
            raise RuntimeError(
                "Regression set failure. Please look at pipeline logs to see the results"
            )
    return evaluation_results


@sematic.func
def run_and_evaluate_results(
    scenario_set: ScenarioSet,
    cache_key: str,
    run_uuid: str,
    perception_pipeline_config: PerceptionPipelineConfig,
    experiment_config_path: Optional[str],
    pipeline_setup: PipelineSetup,
    perception_runner_context: PerceptionRunnerContext,
) -> EvaluationResults:
    """Run scenario set and evaluate results

    Args:
        scenario_set (ScenarioSet): The set of scenarios to run inference on,
        along with a name for the grouping of these scenarios.
        cache_key (str): A unique key that will control usage of the results cache for output of the
        perception node. All results will be cached under the given cache key, so
        changing the key essentially creates a new, empty cache.
        run_uuid (str): A unique way of identifying a new run; the results will be
        stored using this run_uuid.
        It is not cached however and a new run with same uuid will overwrite the results of
        previous run.
        perception_pipeline_config (PerceptionPipelineConfig): Configurations for how the
        perception portion of the pipeline should behave. See the documentation for
        core.infra.sematic.perception.graph_config.PerceptionPipelineConfig
        for details on parameters
        experiment_config_path (Optional[str]): Path to YAML for the experiment config.
        pipeline_setup (PipelineSetup): setup for pipeline
        perception_runner_context (dict): context object for the develop runner

    Returns:
        EvaluationResults: _description_
    """
    evaluation_results = pipeline(
        scenario_set,
        cache_key,
        run_uuid,
        perception_pipeline_config,
        experiment_config_path,
        pipeline_setup=PipelineSetup(),
        perception_runner_context=perception_runner_context,
    )
    return evaluate_results(evaluation_results)


def get_active_cameras() -> List:
    """Get currently active cameras from portal

    Returns:
        List: List of cameras
    """

    with PerceptionPortalSession("PROD") as perception_portal_session:
        response = perception_portal_session.session.post(
            f"{perception_portal_session.host}/graphql/",
            json={"query": ACTIVE_CAMERAS},
            headers=perception_portal_session.headers,
        )
        active_cameras = [
            camera["uuid"]
            for camera in json.loads(response.text)["data"]["cameras"]
            if camera["zone"]["isActive"]
        ]
    return active_cameras


def check_if_valid_incidents_in_portal(incident_camera_list) -> bool:
    """Check for a camera and incident if a valid incident exists in portal

    Args:
        incident_camera_list (List): list of incidents and camera

    Raises:
        RuntimeError: if unable to query portal

    Returns:
        bool: if valid incident exists in portal
    """
    valid_incidents_in_portal = []
    variables = {
        "fromUtc": datetime.strptime("2023-01-01", "%Y-%m-%d").isoformat(),
        "toUtc": None,
        "incidentTypeFilter": None,
        "feedbackType": "valid",
        "cameraUuid": None,
        "organizationKey": None,
        "zoneKey": None,
    }
    with PerceptionPortalSession("PROD") as perception_portal_session:
        for camera, incident in incident_camera_list:
            variables["cameraUuid"] = camera
            variables["incidentTypeFilter"] = incident
            response = perception_portal_session.session.post(
                f"{perception_portal_session.host}/graphql/",
                json={
                    "query": FILTERED_RAW_INCIDENTS_QUERY,
                    "variables": variables,
                },
                headers=perception_portal_session.headers,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Query failed: {response.reason}")
            result_json = json.loads(response.text)["data"]["integrations"][
                "filteredRawIncidents"
            ]
            if result_json["edges"]:
                valid_incidents_in_portal.append((incident, camera))

    return valid_incidents_in_portal


def get_camera_uuid_model_map(active_cameras: list) -> Tuple[dict, set]:
    """
    Get a map of camera uuid to models to test

    Args:
        active_cameras (list): list of active cameras

    Returns:
        dict: map of camera uuid to model type
    """
    cameras_model_map = {}
    all_incidents = set()
    list_of_cameras_file = "configs/cameras/cameras"
    with open(list_of_cameras_file, encoding="utf-8") as cameras_file:
        cameras = []
        for line in cameras_file:
            cameras.append(line.rstrip("\n"))
        for config_path in cameras:
            config = load_yaml_with_jinja(config_path)
            if config["camera_uuid"] not in active_cameras:
                continue
            models = set()
            for state_machine in config["incident"][
                "state_machine_monitors_requested"
            ]:
                if incident_to_model_type.get(state_machine):
                    models.update(incident_to_model_type[state_machine])
                else:
                    logger.info(f"No models requested for {state_machine}")

            # check for uncovered incidents
            incident_machines_to_check = (
                set(config["incident"]["state_machine_monitors_requested"])
                - uncovered_incident_machines
            )

            incidents_for_camera = list(
                {
                    monitor_incident_map[state_machine]
                    for state_machine in incident_machines_to_check
                }
            )
            cameras_model_map[config["camera_uuid"]] = models
            all_incidents.update(incidents_for_camera)

        return cameras_model_map, all_incidents


def get_coverage_metrics(
    live_cameras_model_map_before: dict,
    all_incidents_before: list,
    live_cameras_model_map: dict,
    all_incidents: list,
) -> CoverageResults:
    """Get coverage metrics for regression sets

    Args:
        live_cameras_model_map_before (dict): live cameras to model map before
        checking scenario set
        all_incidents_before (list): all incidents before checking scenario set
        live_cameras_model_map (dict): live cameras to model map after removing
        incidents covered in scenario set
        all_incidents (list): all incidents after removing incidents covered in  scenario set

    Returns:
        CoverageResults: coverage results
    """
    incidents_coverage = (
        (len(all_incidents_before) - len(all_incidents))
        * 100
        / len(all_incidents_before)
    )
    camera_model_incidents_to_cover = 0
    for camera in live_cameras_model_map_before:
        camera_model_incidents_to_cover += len(
            live_cameras_model_map_before[camera]
        )
    camera_model_incidents_not_covered = 0
    for camera in live_cameras_model_map:
        camera_model_incidents_not_covered += len(
            live_cameras_model_map[camera]
        )

    camera_model_coverage = (
        (camera_model_incidents_to_cover - camera_model_incidents_not_covered)
        * 100
        / camera_model_incidents_to_cover
    )
    per_camera_coverage = {}
    for camera in live_cameras_model_map_before:
        if live_cameras_model_map_before[camera]:
            per_camera_coverage[camera] = (
                (
                    len(live_cameras_model_map_before[camera])
                    - len(live_cameras_model_map[camera])
                )
                * 100
                / len(live_cameras_model_map_before[camera])
            )

    model_camera_count = {}
    for _, models in live_cameras_model_map_before.items():
        for model in models:
            if model not in model_camera_count:
                model_camera_count[model] = {
                    "before_coverage": 0,
                    "after_coverage": 0,
                }
            model_camera_count[model]["before_coverage"] += 1
    for _, models in live_cameras_model_map.items():
        for model in models:
            model_camera_count[model]["after_coverage"] += 1
    per_model_coverage = {}
    for model, results in model_camera_count.items():
        per_model_coverage[model] = (
            (results["before_coverage"] - results["after_coverage"])
            * 100
            / results["before_coverage"]
        )

    return CoverageResults(
        incidents_coverage,
        camera_model_coverage,
        per_camera_coverage,
        per_model_coverage,
    )


def slack_notification_for_metrics(coverage_metrics: CoverageResults):
    """Slack notification for coverage metrics

    Args:
        coverage_metrics (CoverageResults): coverage metrics
    """
    webhook = SynchronousWebhookWrapper(get_perception_verbose_sync_webhook())
    block = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Regression test coverage metrics",
            },
        }
    ]
    sections = [
        {
            "type": "mrkdwn",
            "text": "*Incidents coverage:*\n*Camera model coverage:*"
            f"<{os.getenv('BUILDKITE_BUILD_URL', default=None)}"
            "|Look at buildkite for per camera,model coverage>",
        },
        {
            "type": "mrkdwn",
            "text": f"{coverage_metrics.incidents_coverage:.2f}%\n"
            f"{coverage_metrics.camera_model_coverage:.2f}%\n",
        },
    ]
    block.append({"type": "section", "fields": sections})

    webhook.post_message_block(block)


def check_regression_set_complete(scenario_set: ScenarioSet) -> bool:
    """Check if the regression set is missing scenarios

    Args:
        scenario_set (ScenarioSet): set to check in

    Returns:
        bool: if regression set is missing incidents
    """
    active_cameras = get_active_cameras()

    reg_set_camera_incident_map = {}
    for scenario in scenario_set.scenarios:
        if scenario.camera_uuid not in reg_set_camera_incident_map:
            reg_set_camera_incident_map[scenario.camera_uuid] = set()
        reg_set_camera_incident_map[scenario.camera_uuid].update(
            scenario.incidents
        )
    inv_map = {v: k for k, v in monitor_incident_map.items()}
    live_cameras_model_map, all_incidents = get_camera_uuid_model_map(
        active_cameras
    )

    live_cameras_model_map_before = deepcopy(live_cameras_model_map)
    all_incidents_before = deepcopy(all_incidents)
    for camera, incidents in reg_set_camera_incident_map.items():
        if camera in live_cameras_model_map:
            incidents = [inv_map.get(incident) for incident in incidents]
            for incident in incidents:
                if incident in incident_to_model_type:
                    models = incident_to_model_type.get(incident)
                else:
                    models = [incident]
                for model in models:
                    if model in live_cameras_model_map[camera]:
                        live_cameras_model_map[camera].remove(model)
                if monitor_incident_map[incident] in all_incidents:
                    all_incidents.remove(monitor_incident_map[incident])

    coverage_metrics = get_coverage_metrics(
        live_cameras_model_map_before,
        all_incidents_before,
        live_cameras_model_map,
        all_incidents,
    )
    slack_notification_for_metrics(coverage_metrics)
    logger.info(f"CoverageMetrics {coverage_metrics}")
    return (
        coverage_metrics.camera_model_coverage == 100
        and coverage_metrics.incidents_coverage == 100
    )
