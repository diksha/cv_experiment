import os
import re
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import boto3
import mergedeep
import yaml
from botocore.exceptions import ClientError
from loguru import logger

from core.execution.utils.graph_config_builder import GraphConfigBuilder
from core.incident_machine.machines.bumpcap import BumpCapViolationMachine
from core.incident_machine.machines.hardhat import HardHatViolationMachine
from core.incidents.utils import CameraConfig, CameraConfigError
from core.structs.actor import HeadCoveringType
from core.structs.scenario import Scenario
from core.utils.logging.list_indented_yaml_dumper import ListIndentedDumper
from core.utils.yaml_jinja import load_yaml_with_jinja

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)
from services.platform.polygon.lib.proto_schema.validate_graph_config_schema import (
    GraphConfigValidator,
)

# trunk-ignore-end(pylint/E0611)


def push_polygon_configs_to_s3(
    camera_config_path: str,
    environment: str,
    graph_config: dict,
    s3_client: boto3.client = None,
) -> None:
    """This function attempts to retrieve the polygon override file,
    merge it with the graph config from the environment, and then saves the output to s3.

    Args:
        camera_config_path (str): path to config yaml file for the camera
        environment (str): development/staging/production/etc
        graph_config (dict): the config generated from the production runner
        s3_client: (boto3.client): client connection to s3
    """

    if s3_client is None:
        s3_client = boto3.client("s3")

    polygon_override_path = generate_polygon_override_key(camera_config_path)
    polygon_bucket_name = f"voxel-{environment}-polygon-graph-configs"

    polygon_override_exists = True
    try:
        s3_client.head_object(
            Bucket=polygon_bucket_name,
            Key=polygon_override_path,
        )
    except ClientError as exception:
        polygon_override_exists = False
        if exception.response["ResponseMetadata"]["HTTPStatusCode"] == 404:
            logger.warning(
                f"[safe to ignore] file {polygon_override_path} does not exist in bucket "
                f"{polygon_bucket_name}"
            )
        else:
            logger.warning(
                # trunk-ignore(pylint/C0301)
                f"[safe to ignore] exception thrown when checking for key {polygon_override_path} in bucket {polygon_bucket_name}: {str(exception)}"
            )

    validator = GraphConfigValidator()
    if polygon_override_exists:
        full_config = load_config_with_polygon_overrides(
            camera_config_path,
            polygon_override_path,
            polygon_bucket_name,
        )
        validator.validate_dict_against_graph_config_schema(full_config)

    validator.validate_dict_against_graph_config_schema(graph_config)

    try:
        s3_client.put_object(
            Bucket=polygon_bucket_name,
            Key=_generate_graph_config_key(camera_config_path),
            Body=yaml.dump(graph_config, Dumper=ListIndentedDumper),
        )
    except s3_client.exceptions.NoSuchBucket:
        logger.warning(
            f"[safe to ignore] Polgyon bucket {polygon_bucket_name} does not exist"
        )
    # trunk-ignore(pylint/W0718)
    except Exception as exception:
        logger.warning(
            "[safe to ignore] Exception caught when trying to push a Polygon override to s3: "
            f"{str(exception)}"
        )


def generate_polygon_override(
    camera_config_path: str,
    s3_override_key: str,
    s3_bucket: str,
    s3_client: boto3.client,
):
    """Read from s3 and overlay the override file on top of a given camera config

    Args:
        camera_config_path (str): path to config yaml file for the camera
        s3_override_key (str): path to the override file within s3
        s3_bucket (str): the name of the bucket
        s3_client: (boto3.client): client connection to s3

    Raises:
        ValueError: if camera_config_path is a file that must not be overridden

    Returns:
        dict: the resulting layered config
    """

    files_that_should_not_change = [
        "default.yaml",
        "development.yaml",
        "production.yaml",
    ]

    if camera_config_path.endswith(tuple(files_that_should_not_change)):
        raise ValueError(
            f"camera_config_path is {camera_config_path}, which is a file that cannot be overridden"
        )

    response = s3_client.get_object(
        Bucket=s3_bucket,
        Key=s3_override_key,
    )
    returned_body = response["Body"].read()

    config_builder = GraphConfigBuilder()
    camera_config = load_yaml_with_jinja(camera_config_path)
    polygon_override = yaml.safe_load(returned_body.decode("utf-8"))
    config_builder.apply(camera_config, "camera")
    config_builder.apply(polygon_override, "polygon_override")
    return config_builder.get_config()


def push_polygon_override_to_s3(
    final_layered_config: dict,
    s3_bucket: str,
    s3_client: boto3.client,
    polygon_override_key: str,
):
    """Push the given polygon override dict into s3

    Args:
        final_layered_config: the generated polygon override
        s3_bucket (str): the name of the bucket
        s3_client: (boto3.client): client connection to s3
        polygon_override_key: the file path/key of the polygon override file

    Returns:
        dict: the resulting layered config
    """

    s3_client.put_object(
        Bucket=s3_bucket,
        Key=polygon_override_key,
        Body=yaml.dump(final_layered_config, Dumper=ListIndentedDumper),
    )
    return final_layered_config


def generate_polygon_override_key(camera_config_path: str) -> str:
    """Generate the file path/key of the polygon override file

    Args:
        camera_config_path (str): path to config yaml file for the camera

    Returns:
        str: the resulting path of the layered config
    """
    return f"polygon/{camera_config_path}"


def _generate_graph_config_key(camera_config_path: str) -> str:
    """Generate the file path/key of the stored full graph config

    Args:
        camera_config_path (str): path to config yaml file for the camera

    Returns:
        str: the resulting path of the graph config
    """
    return f"graph-configs/{camera_config_path}"


def load_config_with_polygon_overrides(
    camera_config_path: str,
    s3_override_key: str,
    s3_bucket: str,
    s3_client: boto3.client = None,
):
    """Driver func to generate the polygon override and push to s3

    Args:
        camera_config_path (str): path to config yaml file for the camera
        s3_override_key (str): path to the override file within s3
        s3_bucket (str): the name of the bucket
        s3_client: (boto3.client): client connection to s3

    Returns:
        dict: the resulting layered config
    """

    if s3_client is None:
        s3_client = boto3.client("s3")

    polygon_override = generate_polygon_override(
        camera_config_path, s3_override_key, s3_bucket, s3_client
    )
    polygon_override_key = generate_polygon_override_key(camera_config_path)

    return push_polygon_override_to_s3(
        polygon_override, s3_bucket, s3_client, polygon_override_key
    )


def get_updated_local_graph_config(
    cache_key: str,
    portal: bool,
    run_uuid: str,
    log_key: str,
    graph_config_yaml: str = "configs/graphs/develop.yaml",
) -> dict:
    """
    Overwrites the existing local graph config (develop.yaml) or another
    with the values from the input arguments

    Args:
        cache_key (str): the cache key commandline arg
        portal (bool): whether to run local portal
        run_uuid (str): the run uuid to add to the config
        log_key (str): the log key commandline arg
        graph_config_yaml (str, optional): the optional input graph
                              config to use

    Returns:
        dict: the parsed and merged graph config
    """
    cache_key = cache_key.strip("/")
    log_key = log_key.strip("/")
    logger.info(f"Using Cache Key: {cache_key}")
    logger.info(f"Run_uuid: {run_uuid}")

    cli_updates = {
        "cache_key": cache_key,
        "log_key": log_key,
        "run_uuid": run_uuid,
    }

    # If we are doing portal development locally and running the graph
    # then set to use the emulator.
    if portal:
        cli_updates["state"] = {}
        cli_updates["state"]["publisher"] = {
            "state_topic": "projects/local-project/topics/voxel-local-state-messages",
            "event_topic": "projects/local-project/topics/voxel-local-event-messages",
            "emulator_host": "127.0.0.1:31002",
        }

    default_config = load_yaml_with_jinja("configs/graphs/default.yaml")
    additional_config = load_yaml_with_jinja(graph_config_yaml)

    config_builder = GraphConfigBuilder()
    config_builder.apply(default_config, "default")
    config_builder.apply(additional_config, graph_config_yaml.split("/")[-1])
    config_builder.apply(cli_updates, "command_line_args")

    return config_builder.get_config()


def get_camera_uuid_file_map():
    rootdir = os.path.join(os.getcwd(), "configs/cameras")
    camera_uuid_file_map = {}
    for subdir, _dirs, files in os.walk(rootdir):
        for file in files:
            if "experimental" not in subdir and "yaml" in file:
                yaml_parsed = load_yaml_with_jinja(os.path.join(subdir, file))
                if "camera_uuid" in yaml_parsed:
                    camera_uuid_file_map[
                        yaml_parsed["camera_uuid"]
                    ] = os.path.join(subdir, file)
    return camera_uuid_file_map


def get_merged_graph_config_for_camera_uuid(
    graph_config: dict, experiment_config_file: str, camera_uuid: str
) -> dict:
    """
    The merged graph config for the camera uuid

    Args:
        graph_config (dict): original graph config
        experiment_config_file (str): the experiment file to override defaults
        camera_uuid (str): the camera uuid to pull the config for

    Returns:
        dict: the merged graph config
    """
    experiment_config = (
        load_yaml_with_jinja(experiment_config_file)
        if experiment_config_file
        else {}
    )
    camera_uuid_config_file = f"configs/cameras/{camera_uuid}.yaml"
    camera_config = load_yaml_with_jinja(camera_uuid_config_file)
    # oof: mergedeep does this in place
    mergedeep.merge(camera_config, graph_config)
    for key in experiment_config.keys():
        pattern = re.compile(key)
        if pattern.match(camera_uuid):
            mergedeep.merge(camera_config, experiment_config[key])
    return camera_config


def get_scenario_graph_configs_from_scenarios(
    graph_config,
    experiment_config,
    scenarios: List,
    video_uuids_to_include: List,
    enable_video_writer: bool = False,
) -> Tuple[List, List]:
    # Filter to run only on the video uuids if provided.
    if video_uuids_to_include:
        video_uuids_to_include_set = set(video_uuids_to_include)
        scenarios = [
            scenario
            for scenario in scenarios
            if scenario["video_uuid"] in video_uuids_to_include_set
        ]

    graph_configs = []
    all_scenarios = []
    develop_prod_map = {}
    camera_uuid_file_map = get_camera_uuid_file_map()
    for scenario in scenarios:
        if scenario["camera_uuid"] in develop_prod_map:
            camera_config = develop_prod_map[scenario["camera_uuid"]]
        else:
            file_for_camera = camera_uuid_file_map[scenario["camera_uuid"]]
            if not file_for_camera:
                raise RuntimeError(
                    f'Unable to get yaml file for camera uuid{scenario["camera_uuid"]}'
                )
            with open(
                file_for_camera,
                encoding="utf-8",
            ) as stream:
                camera_config = yaml.safe_load(stream)
            mergedeep.merge(camera_config, graph_config)
            if experiment_config:
                for key in experiment_config.keys():
                    pattern = re.compile(key)
                    if pattern.match(scenario["camera_uuid"]):
                        mergedeep.merge(camera_config, experiment_config[key])
            develop_prod_map[scenario["camera_uuid"]] = camera_config
        camera_config_to_append = deepcopy(camera_config)
        camera_config_to_append["video_uuid"] = scenario["video_uuid"]
        camera_config_to_append["enable_video_writer"] = enable_video_writer
        graph_configs.append(camera_config_to_append)
        scenario["config"] = camera_config_to_append
        all_scenarios.append(scenario)

    return graph_configs, all_scenarios


def get_scenario_graph_configs_from_file(
    graph_config,
    experiment_config_path: str,
    config_path: str,
    video_uuids_to_include: List,
    enable_video_writer: bool = False,
) -> Tuple[List, List]:
    config_dict = load_yaml_with_jinja(config_path)
    if "scenario_for_incidents" in config_dict:
        for scenario in config_dict["scenarios"]:
            scenario["scenario_for_incidents"] = config_dict[
                "scenario_for_incidents"
            ]
    experiment_config = None
    if experiment_config_path:
        experiment_config = load_yaml_with_jinja(experiment_config_path)
    return get_scenario_graph_configs_from_scenarios(
        graph_config,
        experiment_config,
        config_dict["scenarios"],
        video_uuids_to_include,
        enable_video_writer=enable_video_writer,
    )


def get_scenario_graph_configs_from_scenario_structs(
    cache_key: str,
    run_uuid: str,
    scenarios: List[Scenario],
    video_uuid_filter: Optional[List[str]],
    experiment_config_path: Optional[str],
) -> List[Any]:
    """Get configurations for a core.execution.graphs.develop.DevelopGraph execution

    Args:
        cache_key: A key to control storage and retrieval of cache results
        run_uuid: A unique id for this execution
        scenarios: The scenarios that graph configs should be obtained for, as Scenario
            struct instances
        video_uuid_filter: If this is not None OR empty, only videos with
            the provided UUIDs will have perception executed on them
        experiment_config_path: Path to the YAML experimental configuration for the scenario set run

    Returns:
        A list of configs that can be used to execute a develop graph with
        DevelopGraph(config).execute()
    """
    experiment_config = None
    if experiment_config_path:
        experiment_config = load_yaml_with_jinja(experiment_config_path)
    graph_config = get_updated_local_graph_config(
        cache_key,
        portal=False,
        run_uuid=run_uuid,
        log_key="",
    )
    (scenario_graph_configs, _,) = get_scenario_graph_configs_from_scenarios(
        graph_config,
        experiment_config=experiment_config,
        scenarios=[asdict(s) for s in scenarios],
        video_uuids_to_include=video_uuid_filter,
    )
    return scenario_graph_configs


def get_should_generate_cooldown_incidents_from_config(config: dict) -> bool:
    """
    Returns whether we should generate cooldown incidents or not in the incident
    writer based on the incoming graph config

    Args:
        config (dict): the graph config

    Returns:
        bool: whether to generate cooldown incidents
    """
    incident_config = config.get("incident", {})
    return bool(incident_config.get("should_generate_on_cooldown", False))


def validate_door_id(camera_uuid: str, annotations: dict) -> bool:
    """Validates whether the door id in the annotations is consistent with the door id
    in the camera config

    Note: We cannot add iou based checking for scenarios where the camera angle changes

    Args:
        camera_uuid (str): unique identifier for the camera name
        annotations (dict): contains the annotations in dict & added to django db

    Returns:
        bool: True if the door id in the annotations is consistent with the door id
        in the camera config
    """
    try:
        camera_config = CameraConfig(camera_uuid, 1, 1)
    except CameraConfigError:
        return True
    annotations_door_ids = [
        door["door_id"] for door in annotations.get("doors")
    ]
    camera_config_door_ids = [door.door_id for door in camera_config.doors]
    next_door_id = camera_config.next_door_id
    for door_id in annotations_door_ids:
        if door_id not in camera_config_door_ids and door_id < next_door_id:
            logger.error(
                f"Door {door_id} should be greater than or"
                f"equal to next door id {camera_uuid}"
            )
            return False

    for door in camera_config.doors:
        if door.door_id not in annotations_door_ids:
            logger.error(
                f"Door id {door.door_id} not found in annotations for {camera_uuid}"
            )
    return True


def validate_scale_and_graph_config_incidents(
    camera_uuid: str, annotations: dict, convert_to_snake_case: bool
) -> bool:
    """Validates whether the scale annotations and the graph configs for any camera
    are consistent with each other

    Args:
        camera_uuid (str): unique identifier for the camera name
        annotations (dict): contains the annotations in dict & added to django db
        convert_to_snake_case (bool): for scale we convert annotation to incident map
        to snake case

    Returns:
        bool: True if the scale annotations and match graph configs

    Raises:
        RuntimeError: If the door id is not consistent with the door id
        in the camera or graph config not found for camera uuid
    """
    if not validate_door_id(camera_uuid, annotations):
        raise RuntimeError(f"Door id not validated for camera: {camera_uuid}")
    is_valid = True
    camera_uuid_incidents = {}
    config_path = f"configs/cameras/{camera_uuid}.yaml"
    annotations = {key for key, value in annotations.items() if value}
    # Ignore annotations not affecting an incident type directly
    annotations -= set(["actionable_regions", "actionableRegions"])
    camera_config_incident_map = {
        "doors": [
            "open_door",
            "door_violation",
            "door_intersection",
            "piggyback",
        ],
        "drivingAreas": ["parking"],
        "intersections": ["intersection", "door_intersection"],
        "endOfAisles": ["no_stop_at_aisle_end"],
        "noPedestrianZones": [
            "no_ped_zone",
            "n_person_ped_zone",
        ],
        "motionDetectionZones": ["production_line_down"],
        "noObstructionRegions": ["obstruction"],
    }
    all_concerned_incidents = {
        incident
        for incidents in camera_config_incident_map.values()
        for incident in incidents
    }
    if convert_to_snake_case:
        camera_config_json_to_incident_map = {}
        for key, value in camera_config_incident_map.items():
            camera_config_json_to_incident_map[
                re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
            ] = value
        camera_config_incident_map = camera_config_json_to_incident_map
    if os.path.exists(config_path):
        config = load_yaml_with_jinja(config_path)
        camera_uuid_incidents = set(
            config["incident"]["state_machine_monitors_requested"]
        )
        # Ignore incidents not requiring camera config
        camera_uuid_incidents = {
            camera_uuid_incident
            for camera_uuid_incident in camera_uuid_incidents
            if camera_uuid_incident in all_concerned_incidents
        }
        iteration_annotation = deepcopy(annotations)
        for annotation in iteration_annotation:
            if bool(
                set(camera_uuid_incidents)
                & set(camera_config_incident_map[annotation])
            ):
                annotations.remove(annotation)
            camera_uuid_incidents -= set(
                camera_config_incident_map[annotation]
            )

        if annotations or camera_uuid_incidents:
            warning_message = (
                f"{camera_uuid}: Additional scale_annotations: {annotations}, "
                f"No annotations for {camera_uuid_incidents}"
            )
            is_valid = False
            # We really want annotations for each incident type
            if camera_uuid_incidents:
                logger.error(warning_message)
            else:
                logger.warning(warning_message)
    else:
        raise RuntimeError(f"Camera config not found for {camera_uuid}")
    return is_valid


def get_gpu_runtime_from_graph_config(config: dict) -> GpuRuntimeBackend:
    """
    Returns the gpu runtime from the graph config

    Args:
        config (dict): the graph config

    Returns:
        GpuRuntimeBackend: the gpu runtime
    """
    _default_runtime = GpuRuntimeBackend.GPU_RUNTIME_BACKEND_LOCAL
    runtime = GpuRuntimeBackend.Value(
        config.get("gpu_runtime", {}).get(
            "runtime", GpuRuntimeBackend.Name(_default_runtime)
        )
    )
    if runtime == GpuRuntimeBackend.GPU_RUNTIME_BACKEND_UNSPECIFIED:
        runtime = _default_runtime
    return runtime


def get_head_covering_type_from_graph_config(
    config: Dict[str, object]
) -> Optional[HeadCoveringType]:
    """
    Gets the head covering type perception is classifying
    based off of what incident machine is enabled. Hard Hat
    is prioritized over covered head for legacy behavior.
    Args:
        config (dict): the graph config
    Returns:
        HeadCoveringType: the head covering type we are classifying
    """
    incident_machines_requested = config.get("incident", {}).get(
        "state_machine_monitors_requested", []
    )
    if HardHatViolationMachine.NAME in incident_machines_requested:
        return HeadCoveringType.HARD_HAT
    if BumpCapViolationMachine.NAME in incident_machines_requested:
        return HeadCoveringType.COVERED_HEAD
    return None
