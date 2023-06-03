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
import os
import warnings
from copy import deepcopy

import pandas as pd
import yaml
from mergedeep import merge
from termcolor import colored

from core.deployments.portal_query_helpers import (
    does_camera_exist,
    does_organization_exist,
    does_zone_exist_for_organization,
    generate_camera_for_organization_zone,
    generate_organization,
    generate_zone_for_organization,
)
from core.labeling.tools.sync_camera_config import sync_camera_config
from core.utils.logging.list_indented_yaml_dumper import ListIndentedDumper

cprt = "\
#\n\
# Copyright 2020-2021 Voxel Labs, Inc.\n\
# All rights reserved.\n\
#\n\
# This document may not be reproduced, republished, distributed, transmitted,\n\
# displayed, broadcast or otherwise exploited in any manner without the express\n\
# prior written permission of Voxel Labs, Inc. The receipt or possession of this\n\
# document does not convey any rights to reproduce, disclose, or distribute its\n\
# contents, or to manufacture, use, or sell anything that it may describe, in\n\
# whole or in part.\n\
#\n\
#\n\
"

K_DEFAULT_HAT_MIN_PIXEL_AREA = 2000
K_DEFAULT_VEST_MIN_PIXEL_AREA = 800

required_scoping_fields = [
    "camera_uuid",
    "camera_arn",
    "kinesis_url",
    "camera_config",
    "bad_posture",
    "door_intersection",
    "door_violation",
    "hard_hat",
    "intersection",
    "no_ped_zone",
    "no_stop_at_aisle_end",
    "open_door",
    "overreaching",
    "parking",
    "piggyback",
    "safety_vest",
    "spill",
    "motion_detection",
    "detector_tracker_model",
    "pose_model",
    "lift_classifier",
    "reach_classifier",
    "reach_classifier_type",
    "door_classifier",
    "hat_classification_by_detection",
    "hat_classifier",
    "hat_detr",
    "vest_classifier",
    "vest_classifier_type",
    "open_door_time",
    "no_parking_area_time",
    "no_ped_zone_time",
    "prod_line_time",
]


def validate_camera_scoping(all_camera_scoping):
    for field in required_scoping_fields:
        if field not in all_camera_scoping:
            raise RuntimeError(f"Scoping document missing: {field}")


def run_sync_camera_config():
    return sync_camera_config()


def add_camera_info(graph_config, camera_scoping):
    graph_config["camera_uuid"] = camera_scoping["camera_uuid"]
    graph_config["camera"]["arn"] = camera_scoping["camera_arn"]
    org = camera_scoping["camera_uuid"].split("/")[0]
    graph_config["publisher"]["organization_key"] = org.upper()


def add_model_path(graph_config, camera_scoping):
    if pd.isna(camera_scoping["detector_tracker_model"]):
        raise RuntimeError("Graph config requires a detector tracker model")
    graph_config["perception"]["detector_tracker"][
        "model_path"
    ] = camera_scoping["detector_tracker_model"]
    if not pd.isna(camera_scoping["pose_model"]):
        graph_config["perception"]["pose"]["enabled"] = True
        graph_config["perception"]["pose"]["model_path"] = camera_scoping[
            "pose_model"
        ]
    if not pd.isna(camera_scoping["lift_classifier"]):
        graph_config["perception"]["lift_classifier"]["enabled"] = True
        graph_config["perception"]["lift_classifier"][
            "model_path"
        ] = camera_scoping["lift_classifier"]
    if not pd.isna(camera_scoping["reach_classifier"]) and not pd.isna(
        camera_scoping["reach_classifier_type"]
    ):
        graph_config["perception"]["reach_classifier"]["enabled"] = True
        graph_config["perception"]["reach_classifier"][
            "model_path"
        ] = camera_scoping["reach_classifier"]
        graph_config["perception"]["reach_classifier"][
            "model_type"
        ] = camera_scoping["reach_classifier_type"]
    if not pd.isna(camera_scoping["door_classifier"]):
        graph_config["perception"]["door_classifier"]["enabled"] = True
        graph_config["perception"]["door_classifier"][
            "model_path"
        ] = camera_scoping["door_classifier"]
        graph_config["perception"]["door_classifier"][
            "model_type"
        ] = "vanilla_resnet"
    if not pd.isna(camera_scoping["hat_classifier"]) and not pd.isna(
        camera_scoping["hat_classification_by_detection"]
    ):
        graph_config["perception"]["hat_classifier"]["enabled"] = True
        graph_config["perception"]["hat_classifier"][
            "min_actor_pixel_area"
        ] = K_DEFAULT_HAT_MIN_PIXEL_AREA
        if not pd.isna(camera_scoping["hat_detr"]):
            graph_config["perception"]["hat_classifier"][
                "detr_model_path"
            ] = camera_scoping["hat_detr"]
        graph_config["perception"]["hat_classifier"][
            "model_path"
        ] = camera_scoping["hat_classifier"]
        graph_config["perception"]["hat_classifier"][
            "is_classification_by_detection"
        ] = camera_scoping["hat_classification_by_detection"]
    if not pd.isna(camera_scoping["vest_classifier"]) and not pd.isna(
        camera_scoping["vest_classifier_type"]
    ):
        graph_config["perception"]["vest_classifier"]["enabled"] = True
        graph_config["perception"]["vest_classifier"][
            "model_path"
        ] = camera_scoping["vest_classifier"]
        graph_config["perception"]["vest_classifier"][
            "model_type"
        ] = camera_scoping["vest_classifier_type"]
        graph_config["perception"]["vest_classifier"][
            "min_actor_pixel_area"
        ] = K_DEFAULT_VEST_MIN_PIXEL_AREA

    # Enable motion detection zone
    graph_config["perception"]["motion_zone_detection"][
        "enabled"
    ] = camera_scoping["motion_detection"]


def _add_required_ergo_incidents(config, scoping, incident_dict):
    if (
        scoping["bad_posture"]
        and config["perception"]["lift_classifier"]["enabled"]
    ):
        incident_dict["state_machine_monitors_requested"].append("bad_posture")
    if (
        scoping["overreaching"]
        and config["perception"]["reach_classifier"]["enabled"]
    ):
        incident_dict["state_machine_monitors_requested"].append(
            "overreaching"
        )


def _add_required_door_incidents(scoping, incident_dict):
    if scoping["door_intersection"]:
        incident_dict["state_machine_monitors_requested"].append(
            "door_intersection"
        )
    if scoping["door_violation"]:
        incident_dict["state_machine_monitors_requested"].append(
            "door_violation"
        )
    if scoping["open_door"]:
        incident_dict["state_machine_monitors_requested"].append("open_door")
    if scoping["piggyback"]:
        incident_dict["state_machine_monitors_requested"].append("piggyback")


def _add_incident_machines_parameters(graph_config, camera_scoping):
    """This function adds incident machine parameters for open_door, parking, no_ped_zone
    and motion_detection if enabled and filled in the scoping doc.

    Args:
        graph_config (_type_): camera yaml file
        camera_scoping (_type_): camera scoping having all the required fields
    """
    if camera_scoping["open_door"] and not pd.isna(
        camera_scoping["open_door_time"]
    ):
        merge(
            graph_config,
            {
                "incident": {
                    "incident_machine_params": {
                        "open_door": {
                            "max_open_door_s": int(
                                camera_scoping["open_door_time"]
                            )
                        }
                    }
                }
            },
        )
    if camera_scoping["no_ped_zone"] and not pd.isna(
        camera_scoping["no_ped_zone_time"]
    ):
        merge(
            graph_config,
            {
                "incident": {
                    "incident_machine_params": {
                        "no_ped_zone": {
                            "max_ped_zone_duration_s": int(
                                camera_scoping["no_ped_zone_time"]
                            )
                        }
                    }
                }
            },
        )
    if camera_scoping["parking"] and not pd.isna(
        camera_scoping["no_parking_area_time"]
    ):
        merge(
            graph_config,
            {
                "incident": {
                    "incident_machine_params": {
                        "parking": {
                            "max_parked_duration_s": int(
                                camera_scoping["no_parking_area_time"]
                            )
                        }
                    }
                }
            },
        )
    if camera_scoping["motion_detection"] and not pd.isna(
        camera_scoping["prod_line_time"]
    ):
        merge(
            graph_config,
            {
                "incident": {
                    "incident_machine_params": {
                        "production_line_down": {
                            "max_no_motion_detection_s": int(
                                camera_scoping["prod_line_time"]
                            )
                        }
                    }
                }
            },
        )


def add_monitors_and_incident_machines(graph_config, camera_scoping):
    # Relies on detector only
    incidents = graph_config["incident"]
    if camera_scoping["intersection"]:
        incidents["state_machine_monitors_requested"].append("intersection")
    if camera_scoping["no_ped_zone"]:
        incidents["state_machine_monitors_requested"].append("no_ped_zone")
    if camera_scoping["no_stop_at_aisle_end"]:
        incidents["state_machine_monitors_requested"].append(
            "no_stop_at_aisle_end"
        )
    if camera_scoping["parking"]:
        incidents["state_machine_monitors_requested"].append("parking")
    # Relies on ergo models
    if graph_config["perception"]["pose"]["enabled"]:
        _add_required_ergo_incidents(graph_config, camera_scoping, incidents)
    # Relies on door model
    if graph_config["perception"]["door_classifier"]["enabled"]:
        _add_required_door_incidents(camera_scoping, incidents)
    # Relies on PPE models
    if (
        camera_scoping["hard_hat"]
        and graph_config["perception"]["hat_classifier"]["enabled"]
    ):
        incidents["state_machine_monitors_requested"].append("hard_hat")
    if (
        camera_scoping["safety_vest"]
        and graph_config["perception"]["vest_classifier"]["enabled"]
    ):
        incidents["state_machine_monitors_requested"].append("safety_vest")
    # Relies on no models
    if camera_scoping["spill"]:
        incidents["state_machine_monitors_requested"].append("random_spill")
    incidents["state_machine_monitors_requested"] = sorted(
        incidents["state_machine_monitors_requested"]
    )
    if camera_scoping["motion_detection"]:
        incidents["state_machine_monitors_requested"].append(
            "production_line_down"
        )
    _add_incident_machines_parameters(graph_config, camera_scoping)


def check_for_other_updated_configs(updated_camera_configs):
    for camera_uuid in updated_camera_configs.keys():
        if updated_camera_configs[camera_uuid] is None:
            warnings.warn(
                colored(
                    f"{camera_uuid} config missing but is present in db",
                    "red",
                )
            )
            continue
        if updated_camera_configs[camera_uuid]["isUpdated"]:
            warnings.warn(
                colored(
                    f"{camera_uuid} was updated during sync, please create PR with new config or delete config from db",
                    "red",
                )
            )


def sync_camera_config_version_with_portal(
    graph_config, camera_uuid, camera_config_from_portal
):
    if camera_config_from_portal is not None:
        version = camera_config_from_portal["cameraConfigNew"]["version"]
        graph_config["camera"]["version"] = version
    else:
        warnings.warn(
            colored(
                f"Cannot find updated config for {camera_uuid}, skipping",
                "red",
            )
        )


def create_all_graph_configs(
    site_scoping,
    template_graph_config_path,
    output_dir_path,
    updated_camera_configs,
):
    with open(template_graph_config_path, "r", encoding="UTF-8") as file:
        template_graph_config = yaml.safe_load(file)
    for _, camera_scoping_df in site_scoping.iterrows():
        # Create config
        graph_config_dict = deepcopy(template_graph_config)
        add_camera_info(graph_config_dict, camera_scoping_df)
        add_model_path(graph_config_dict, camera_scoping_df)
        add_monitors_and_incident_machines(
            graph_config_dict, camera_scoping_df
        )
        # If syncing config, update version to latest version
        if updated_camera_configs is not None:
            synced_camera_config = updated_camera_configs.pop(
                camera_scoping_df.camera_uuid, None
            )
            sync_camera_config_version_with_portal(
                graph_config_dict,
                camera_scoping_df.camera_uuid,
                synced_camera_config,
            )
        # Write updated yaml
        org, loc, zone, _ = camera_scoping_df.camera_uuid.split("/")
        save_dir = os.path.join(output_dir_path, f"{org}/{loc}/{zone}")
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/cha.yaml", "w", encoding="UTF-8") as file:
            file.write(cprt)
            yaml.dump(graph_config_dict, file, Dumper=ListIndentedDumper)
        graph_config_dict.clear()


def create_portal_cameras_if_required(site_scoping):
    portal_env = "PROD"
    cameras = site_scoping.camera_uuid.to_list()
    org, zone, _, _, = cameras[
        0
    ].split("/")
    org_key = org.upper()
    zone_key = zone.upper()
    if not does_organization_exist(portal_env, org_key):
        org_name = org.lower().replace("_", " ").title()
        generate_organization(portal_env, org_key, org_name)
    if not does_zone_exist_for_organization(portal_env, org_key, zone_key):
        zone_name = zone.lower().replace("_", " ").title()
        generate_zone_for_organization(
            portal_env, org_key, zone_key, zone_name, "site"
        )
    for camera_uuid in cameras:
        _, _, number, _ = camera_uuid.split("/")
        camera_number = int(number)
        if not does_camera_exist(portal_env, camera_uuid):
            camera_name = f"Camera {camera_number}"
            generate_camera_for_organization_zone(
                portal_env, org_key, zone_key, camera_uuid, camera_name
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoping_csv_path", "-csv", type=str, required=True)
    parser.add_argument(
        "--template_graph_config_path", "-t", type=str, required=True
    )
    parser.add_argument("--output_dir", "-out", type=str, required=True)
    parser.add_argument("--create_portal_cameras", action="store_true")
    parser.add_argument("--sync_camera_config", "-sync", action="store_true")
    return parser.parse_args()


def main(args):
    site_scoping_df = pd.read_csv(args.scoping_csv_path)
    validate_camera_scoping(site_scoping_df)
    if args.create_portal_cameras:
        create_portal_cameras_if_required(site_scoping_df)
    updated_portal_camera_configs = None
    if args.sync_camera_config:
        updated_portal_camera_configs = run_sync_camera_config()
    create_all_graph_configs(
        site_scoping_df,
        args.template_graph_config_path,
        args.output_dir,
        updated_portal_camera_configs,
    )
    if updated_portal_camera_configs is not None:
        check_for_other_updated_configs(updated_portal_camera_configs)


if __name__ == "__main__":
    script_arguments = parse_args()
    main(script_arguments)
