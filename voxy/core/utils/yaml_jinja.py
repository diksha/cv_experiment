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
import datetime
import os
import typing
import uuid
from io import StringIO

import jinja2
import yaml
from loguru import logger

from core.metaverse.metaverse import Metaverse
from core.utils.aws_utils import get_value_from_aws_secret_manager
from core.utils.camera_config_utils import (
    get_camera_uuids_from_organization_site,
)


class NamedStringIO(StringIO):
    def __init__(self, string_content, name):
        super().__init__(string_content)
        self.name = name


def jinja_parser(
    name, val_type, default_value=None, required=False, nargs=None
):
    """Jinja Parser.

    Jinja parser to fetch arguments from command line and update the input config.

    Args:
        name (str): name of the argument
        val_type (str): type of the argument, float, str, int
        default_value (str, optional): default value of the argument
        required (bool, optional): whether argument is required
        nargs (str, optional): a different number of command-line arguments with a single action

    Returns:
        Any: attribute
    """

    parser = argparse.ArgumentParser(allow_abbrev=False)
    arg_type = {"str": str, "float": float, "int": int}[val_type]
    if not nargs:
        parser.add_argument(
            f"--{name}",
            type=arg_type,
            default=default_value,
            required=required,
        )
    else:
        parser.add_argument(
            f"--{name}",
            type=arg_type,
            default=default_value,
            required=required,
            nargs=nargs,
        )

    args, _ = parser.parse_known_args()
    return getattr(args, name)


def get_camera_uuids_from_args() -> typing.List[str]:
    """
    Adds `--organization` (required) and `--location` (optional) arguments to the
    argparser and parses these to produce a set of valid camera uuids

    Returns:
        typing.List[str]: the list of camera uuids in the format
                            [`organization/location/zone/channel`,...]
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--organization",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--location",
        type=str,
        required=False,
        default=None,
    )
    args, _ = parser.parse_known_args()
    return get_camera_uuids_from_organization_site(
        organization=args.organization, location=args.location
    )


def items_from_file(filepath):
    return open(filepath).read().splitlines()


def items_from_metaverse(query_list):
    metaverse = Metaverse()
    items = []
    for query in query_list:
        data_collection_query = """query get_data_collection_path_contains($path: String) {
            data_collection_path_contains(path: $path) {
                path
            }
        }
        """
        qvars = {"path": query}
        data_collection_result = metaverse.schema.execute(
            data_collection_query, qvars
        )
        logger.info(
            (
                f"Data collection result "
                f'{data_collection_result.data["data_collection_path_contains"]}'
            )
        )
        items.extend(
            [
                data_collection_result.data["data_collection_path_contains"][
                    i
                ]["path"]
                for i, _ in enumerate(
                    data_collection_result.data[
                        "data_collection_path_contains"
                    ]
                )
            ]
        )
    return items


def extract_scenarios_from_files(paths):
    scenarios = []
    paths = paths if isinstance(paths, (list, tuple)) else [paths]
    for path in paths:
        if load_yaml_with_jinja(path)["scenarios"]:
            parsed_file = load_yaml_with_jinja(path)
            for scenario in parsed_file["scenarios"]:
                scenario["scenario_for_incidents"] = parsed_file[
                    "scenario_for_incidents"
                ]
                scenarios.append(scenario)
    return scenarios


def generate_scenarios_video_uuids(paths):
    video_uuids = []
    paths = paths if isinstance(paths, (list, tuple)) else [paths]
    for path in paths:
        scenarios = load_yaml_with_jinja(path)["scenarios"]
        video_uuids.extend([scenario["video_uuid"] for scenario in scenarios])
    return video_uuids


def generate_uuid_with_prefix(prefix):
    return f"{prefix}{str(uuid.uuid4())[:16]}"


def generate_small_uuid_with_prefix(prefix):
    return f"{prefix}{str(uuid.uuid4())[:4]}"


def generate_current_date_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d-%H-%M-%S")


def resolve_jinja_config(config_path: str, **kwargs) -> NamedStringIO:
    """Resolves the jinja config given config path and extra args:

    Example Usage of args:
      config = load_yaml_with_jinja(config_path, key="value")
      In yaml file:
        key: {{key}}

    Args:
        config_path (str): Path of the configuration
        **kwargs: Extra arguments for jinja

    Returns:
        NamedStringIO: Named byte string from yaml
    """

    def load_template(name):
        return open(os.path.join(os.path.split(config_path)[0], name)).read()

    jinja_environment = jinja2.Environment(
        loader=jinja2.FunctionLoader(load_template)
    )
    jinja_environment.globals["JINJA_PARSER"] = jinja_parser
    jinja_environment.globals["ITEMS_FROM_FILE"] = items_from_file
    jinja_environment.globals["ITEMS_FROM_METAVERSE"] = items_from_metaverse
    jinja_environment.globals[
        "JINJA_SECRET_FROM_AWS"
    ] = get_value_from_aws_secret_manager
    jinja_environment.globals[
        "EXTRACT_SCENARIOS_FROM_FILES"
    ] = extract_scenarios_from_files
    jinja_environment.globals[
        "GENERATE_SCENARIO_VIDEO_UUIDS"
    ] = generate_scenarios_video_uuids
    jinja_environment.globals[
        "GENERATE_UUID_WITH_PREFIX"
    ] = generate_uuid_with_prefix
    jinja_environment.globals[
        "GENERATE_CURRENT_DATE_TIME"
    ] = generate_current_date_time
    jinja_environment.globals[
        "GENERATE_SMALL_UUID_WITH_PREFIX"
    ] = generate_small_uuid_with_prefix
    jinja_environment.globals[
        "JINJA_GET_CAMERA_UUIDS_PARSER"
    ] = get_camera_uuids_from_args

    template = jinja_environment.from_string(open(config_path).read())
    resolved_config_string = template.render(
        ENV=os.environ, jinja_filename=config_path, **kwargs
    )
    return NamedStringIO(resolved_config_string, name=config_path)


def load_yaml_with_jinja(config_path: str, **kwargs) -> dict:
    """Dictionary of configuration given yaml

    Args:
        config_path (str): path of config file
        **kwargs: Extra arguments for jinja

    Returns:
        dict: dict representation of yaml
    """

    return yaml.safe_load(resolve_jinja_config(config_path, **kwargs))
