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

import logging
import os
import uuid

import sentry_sdk
from loguru import logger
from sentry_sdk.integrations.logging import BreadcrumbHandler, EventHandler

from core.execution.graphs.production import ProductionGraph
from core.execution.utils.graph_config_builder import GraphConfigBuilder
from core.execution.utils.graph_config_utils import push_polygon_configs_to_s3
from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)
from core.utils.logger import ALL_LOGGING_LEVELS, configure_logger
from core.utils.yaml_jinja import load_yaml_with_jinja

ENVIRONMENT_DEV = "development"
ENVIRONMENT_PROD = "production"
ENVIRONMENT_STAGING = "staging"
RUNTIME_ENVIRONMENTS = [ENVIRONMENT_DEV, ENVIRONMENT_PROD, ENVIRONMENT_STAGING]


def _generate_run_uuid() -> str:
    release_tag = os.getenv("IMAGE_TAG") if os.getenv("IMAGE_TAG") else ""
    salt = str(uuid.uuid4())
    return f"{release_tag}:{salt}"


def _configure_logging(logging_level: str, serialize_logs: bool) -> None:
    configure_logger(level=logging_level, serialize=serialize_logs)

    # Add integration to pass in loguru logs to sentry.
    if os.getenv("SENTRY_DSN"):
        logger.add(BreadcrumbHandler(level=logging.ERROR), level=logging.ERROR)
        logger.add(EventHandler(level=logging.ERROR), level=logging.ERROR)

    sentry_sdk.init(
        traces_sample_rate=1.0,
    )


class ProductionRunner:
    def __init__(
        self,
        camera_config_path: str,
        env: str,
        logging_level: str,
        serialize_logs: bool,
        triton_server_url: str,
    ):
        self._perception_runner_context = (
            PerceptionRunnerContext(_triton_server_url=triton_server_url)
            if triton_server_url
            else PerceptionRunnerContext()
        )
        self._env = env
        self._camera_config_path = camera_config_path

        if env not in RUNTIME_ENVIRONMENTS:
            raise ValueError(
                f"Invalid environment {env}. Must be one of {RUNTIME_ENVIRONMENTS}"
            )

        if logging_level not in ALL_LOGGING_LEVELS:
            raise ValueError(
                f"Invalid logging level {logging_level}. Must be one of {ALL_LOGGING_LEVELS}"
            )

        # Configure Logger
        _configure_logging(logging_level, serialize_logs)

        # Create a unique run id for this run
        self._run_uuid = _generate_run_uuid()

        # Load Graph Configuration
        self._graph = None
        self._graph_config_paths = self._load_graph_config_paths()
        self._graph_config = self._load_graph_config()
        self._push_polygon_configs_to_s3()
        self._add_run_uuid_to_config()
        self._graph = self._init_graph()

    def _load_graph_config_paths(self) -> dict:
        return {
            "default": "configs/graphs/default.yaml",
            "camera": self._camera_config_path,
            "environment": f"configs/graphs/production/environment/{self._env}.yaml",
        }

    def _load_graph_config(self) -> dict:
        """Load and combine config files from paths provided in args

        Args:
            camera_config_path (str): path to config yaml file for the camera
            env_config_path (str): path to config yaml file for the execution environment

        Returns:
            dict: combined config
        """

        config_builder = GraphConfigBuilder()
        for config_key in ["default", "camera", "environment"]:
            config = load_yaml_with_jinja(self._graph_config_paths[config_key])
            config_builder.apply(config, config_key)

        return config_builder.get_config()

    def _push_polygon_configs_to_s3(self):
        try:
            push_polygon_configs_to_s3(
                self._graph_config_paths["camera"],
                self._env,
                self._graph_config.copy(),
            )
        # trunk-ignore(pylint/W0718)
        except Exception as exception:
            logger.warning(
                f"[safe to ignore] Exception caught when trying to run Polygon: {str(exception)}"
            )

    def _add_run_uuid_to_config(self):
        self._graph_config["run_uuid"] = self._run_uuid

    def _init_graph(self):
        if self._graph_config.get("deprecated", False):
            raise RuntimeError(
                f"Camera config at {self._graph_config_paths['camera']} is deprecated."
            )

        return ProductionGraph(
            self._graph_config, self._env, self._perception_runner_context
        )

    def run(self):
        self._graph.execute()
