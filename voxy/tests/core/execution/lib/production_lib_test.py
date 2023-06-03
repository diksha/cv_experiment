import json
import unittest
from unittest.mock import Mock, patch

from core.execution.lib.production_lib import ProductionRunner
from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)


class ProductionRunnerTest(unittest.TestCase):
    @patch("core.execution.lib.production_lib.push_polygon_configs_to_s3")
    @patch("core.execution.lib.production_lib._generate_run_uuid")
    @patch(
        "core.execution.lib.production_lib.ProductionRunner._load_graph_config_paths"
    )
    @patch("core.execution.lib.production_lib.ProductionGraph")
    def test_runner(
        self,
        mock_production_graph,
        mock_load_graph_config_paths,
        mock_generate_run_uuid,
        mock_push_polygon_configs_to_s3,
    ):
        # Arrange
        environment = "production"
        perception_runner_context = PerceptionRunnerContext()
        test_data_path = "tests/core/execution/lib/testdata"
        camera_config_path = f"{test_data_path}/sample_cha.yaml"
        default_config_path = f"{test_data_path}/sample_default.yaml"
        environment_config_path = (
            f"{test_data_path}/sample_production_env.yaml"
        )

        mock_generate_run_uuid.return_value = "test_run_uuid"

        mock_load_graph_config_paths.return_value = {
            "camera": camera_config_path,
            "default": default_config_path,
            "environment": environment_config_path,
        }

        mock_production_graph_instance = Mock()
        mock_production_graph.return_value = mock_production_graph_instance

        with open(
            f"{test_data_path}/expected_graph_config.json",
            "r",
            encoding="utf-8",
        ) as file:
            expected_graph_config = json.load(file)

        # Act
        production = ProductionRunner(
            camera_config_path=camera_config_path,
            env=environment,
            logging_level="INFO",
            serialize_logs=False,
            triton_server_url="",
        )

        production.run()

        # Assert
        mock_production_graph.assert_called_once_with(
            expected_graph_config, environment, perception_runner_context
        )
        mock_production_graph_instance.execute.assert_called_once_with()

        del expected_graph_config["run_uuid"]
        mock_push_polygon_configs_to_s3.assert_called_once_with(
            camera_config_path,
            environment,
            expected_graph_config,
        )

    @patch("core.execution.lib.production_lib.push_polygon_configs_to_s3")
    @patch("core.execution.lib.production_lib._generate_run_uuid")
    @patch(
        "core.execution.lib.production_lib.ProductionRunner._load_graph_config_paths"
    )
    @patch("core.execution.lib.production_lib.ProductionGraph")
    def test_runner_with_triton_server_url(
        self,
        mock_production_graph,
        mock_load_graph_config_paths,
        mock_generate_run_uuid,
        mock_push_polygon_configs_to_s3,
    ):
        # Arrange
        environment = "production"
        perception_runner_context = PerceptionRunnerContext(
            "triton.server.url"
        )
        test_data_path = "tests/core/execution/lib/testdata"
        camera_config_path = f"{test_data_path}/sample_cha.yaml"
        default_config_path = f"{test_data_path}/sample_default.yaml"
        environment_config_path = (
            f"{test_data_path}/sample_production_env.yaml"
        )

        mock_generate_run_uuid.return_value = "test_run_uuid"

        mock_load_graph_config_paths.return_value = {
            "camera": camera_config_path,
            "default": default_config_path,
            "environment": environment_config_path,
        }

        mock_production_graph_instance = Mock()
        mock_production_graph.return_value = mock_production_graph_instance

        with open(
            f"{test_data_path}/expected_graph_config.json",
            "r",
            encoding="utf-8",
        ) as file:
            expected_graph_config = json.load(file)

        # Act
        production = ProductionRunner(
            camera_config_path=camera_config_path,
            env=environment,
            logging_level="INFO",
            serialize_logs=False,
            triton_server_url="triton.server.url",
        )

        production.run()

        # Assert
        mock_production_graph.assert_called_once_with(
            expected_graph_config, environment, perception_runner_context
        )
        mock_production_graph_instance.execute.assert_called_once_with()

        del expected_graph_config["run_uuid"]
        mock_push_polygon_configs_to_s3.assert_called_once_with(
            camera_config_path,
            environment,
            expected_graph_config,
        )
