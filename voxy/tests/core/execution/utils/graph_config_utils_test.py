# trunk-ignore-all(pylint/C0301)
import json
import unittest
from io import BytesIO

import mock
import yaml
from botocore.response import StreamingBody
from botocore.stub import Stubber

from core.execution.utils.graph_config_builder import GraphConfigBuilder
from core.execution.utils.graph_config_utils import (
    _generate_graph_config_key,
    generate_polygon_override_key,
    get_gpu_runtime_from_graph_config,
    get_head_covering_type_from_graph_config,
    get_scenario_graph_configs_from_file,
    get_should_generate_cooldown_incidents_from_config,
    get_updated_local_graph_config,
    load_config_with_polygon_overrides,
    push_polygon_configs_to_s3,
    validate_door_id,
    validate_scale_and_graph_config_incidents,
)
from core.structs.actor import HeadCoveringType
from core.utils.logging.list_indented_yaml_dumper import ListIndentedDumper
from core.utils.yaml_jinja import load_yaml_with_jinja
from lib.utils.test_utils.aws_client_utils import get_boto_client_with_no_creds

# trunk-ignore-begin(pylint/E0611)
# trunk can't see the generated protos
from protos.perception.graph_config.v1.perception_params_pb2 import (
    GpuRuntimeBackend,
)

# trunk-ignore-end(pylint/E0611)


class GraphConfigUtilsTest(unittest.TestCase):
    def test_load_config_with_polygon_overrides(self) -> None:
        test_bucket = "test-bucket"
        test_layered_result_path = (
            "tests/configs/cameras/test-layered-result.yaml"
        )
        test_override_path = "tests/configs/cameras/test-override.yaml"
        test_camera_config_path = "tests/configs/cameras/test-cha.yaml"

        config_builder = GraphConfigBuilder()
        test_layered_result_config = load_yaml_with_jinja(
            test_layered_result_path
        )
        config_builder.apply(
            test_layered_result_config, "graph_config_utils_test"
        )

        body_encoded = json.dumps(test_layered_result_config).encode("utf-8")
        mock_body = StreamingBody(
            BytesIO(body_encoded),
            len(body_encoded),
        )

        expected_get_object_params = {
            "Bucket": test_bucket,
            "Key": test_override_path,
        }
        get_object_response = {"Body": mock_body}

        expected_put_object_params = {
            "Bucket": test_bucket,
            "Body": yaml.dump(
                config_builder.get_config(), Dumper=ListIndentedDumper
            ),
            "Key": generate_polygon_override_key(test_camera_config_path),
        }
        put_object_response = {}

        s3_client = get_boto_client_with_no_creds("s3")
        stubber = Stubber(s3_client)
        stubber.add_response(
            "get_object", get_object_response, expected_get_object_params
        )
        stubber.add_response(
            "put_object", put_object_response, expected_put_object_params
        )
        stubber.activate()

        service_response = load_config_with_polygon_overrides(
            test_camera_config_path,
            test_override_path,
            test_bucket,
            s3_client=s3_client,
        )
        assert service_response == test_layered_result_config

    @mock.patch(
        "core.execution.utils.graph_config_utils.load_config_with_polygon_overrides"
    )
    def test_push_polygon_configs_to_s3(
        self, mock_load_config_with_polygon_overrides
    ) -> None:
        environment = "test"
        test_camera_config_path = "tests/configs/cameras/test-cha.yaml"
        test_camera_config = load_yaml_with_jinja(test_camera_config_path)
        polygon_bucket_name = f"voxel-{environment}-polygon-graph-configs"

        head_object_response = {}
        head_object_expected_params = {
            "Bucket": polygon_bucket_name,
            "Key": generate_polygon_override_key(test_camera_config_path),
        }
        put_object_response = {}
        put_object_expected_params = {
            "Bucket": polygon_bucket_name,
            "Key": _generate_graph_config_key(test_camera_config_path),
            "Body": yaml.dump(test_camera_config, Dumper=ListIndentedDumper),
        }
        s3_client = get_boto_client_with_no_creds("s3")
        stubber = Stubber(s3_client)
        stubber.add_response(
            "head_object", head_object_response, head_object_expected_params
        )
        stubber.add_response(
            "put_object", put_object_response, put_object_expected_params
        )
        stubber.activate()

        mock_load_config_with_polygon_overrides.return_value = {}

        # test the case where no errors are thrown
        push_polygon_configs_to_s3(
            test_camera_config_path, environment, test_camera_config, s3_client
        )

        stubber.add_client_error(
            "head_object",
            service_error_code="NoSuchKey",
            response_meta={"HTTPStatusCode": 404},
        )

        # test the case where the polygon override key does not exist
        push_polygon_configs_to_s3(
            test_camera_config_path, environment, test_camera_config, s3_client
        )

    def test_should_generate_cooldown_incidents(self) -> None:
        """
        Tests the config for whether to generate cooldown incidents
        """
        dummy_config = {}
        self.assertFalse(
            get_should_generate_cooldown_incidents_from_config(dummy_config)
        )
        dummy_config = {"incident": {"foo": "hello", "bar": "world"}}
        self.assertFalse(
            get_should_generate_cooldown_incidents_from_config(dummy_config)
        )
        dummy_config = {"incident": {"should_generate_on_cooldown": True}}
        self.assertTrue(
            get_should_generate_cooldown_incidents_from_config(dummy_config)
        )
        dummy_config = {"incident": {"should_generate_on_cooldown": False}}
        self.assertFalse(
            get_should_generate_cooldown_incidents_from_config(dummy_config)
        )
        dummy_config = {"incident": {"should_generate_on_cooldown": None}}
        self.assertFalse(
            get_should_generate_cooldown_incidents_from_config(dummy_config)
        )

    def test_get_scenario_graph_configs_from_file(
        self,
    ) -> None:
        config_path = "data/unit_test_resources/test_scenario_config.yaml"
        graph_config = get_updated_local_graph_config(
            "cache_key", True, "uuid", ""
        )
        graph_configs, all_scenarios = get_scenario_graph_configs_from_file(
            graph_config,
            None,
            config_path,
            None,
        )
        assert all_scenarios == [
            {
                "camera_uuid": "americold/modesto/0001/cha",
                "incidents": ["DOOR_VIOLATION"],
                "video_uuid": "americold/modesto/0001/cha/5bb70c8b-b007-476e-a0d8-877d17db336c_0000",
                "config": {
                    "camera": {
                        "arn": "arn:aws:kinesisvideo:us-west-2:360054435465:stream/americold-modesto-0001/1652139618645",
                        "version": 3,
                        "fps": 5,
                        "min_frame_difference_ms": 85,
                    },
                    "camera_uuid": "americold/modesto/0001/cha",
                    "incident": {
                        "should_generate_on_cooldown": True,
                        "incident_machine_params": {
                            "parking": {"max_parked_duration_s": 300}
                        },
                        "monitors_requested": [],
                        "state_machine_monitors_requested": [
                            "bad_posture",
                            "intersection",
                            "parking",
                            "overreaching",
                            "door_intersection",
                            "door_violation",
                            "open_door",
                            "piggyback",
                            "spill",
                        ],
                        "dry_run": True,
                        "temp_directory": "/var/tmp/voxel/incidents",
                        "generate_temp_subdirs": True,
                    },
                    "perception": {
                        "detector_tracker": {
                            "actor2class": {"PERSON": 0, "PIT": 1},
                            "height": 480,
                            "model_path": "artifacts_2021-12-06-00-00-00-0000-yolo/best_480_960.engine",
                            "width": 960,
                        },
                        "door_classifier": {
                            "enabled": True,
                            "model_path": "artifacts_2022-10-26_americold_modesto_0001_cha/2022-10-26_americold_modesto_0001_cha.pt",
                            "model_type": "vanilla_resnet",
                        },
                        "hat_classifier": {"enabled": False},
                        "lift_classifier": {
                            "enabled": True,
                            "model_type": "HM",
                            "model_path": "artifacts_lift_classifier_01282022/lift_classifer_01282022.sav",
                        },
                        "pose": {
                            "enabled": True,
                            "model_path": "artifacts_03_21_2023_pose_0630_jit_update/fast_res50_256x192.pt",
                        },
                        "reach_classifier": {
                            "enabled": True,
                            "model_path": "artifacts_03_23_2023_overreaching_model_jit/voxel_ergo_ml_overreaching_2022-05-23-jit.pt",
                            "model_type": "DL",
                        },
                        "vest_classifier": {"enabled": False},
                        "spill": {
                            "enabled": True,
                            "model_path": "artifacts_02_05_2023_spill_generalized_jit/smp-bd6b62dfd5b64dcfb4970102e2c9b2aa-jit.pt",
                            "min_run_time_difference_ms": 15000,
                            "min_pixel_size": 200,
                            "max_consecutive_runs": 1,
                            "post_process_enabled": True,
                            "frame_segment2class": {"UNKNOWN": 0, "SPILL": 1},
                        },
                        "obstruction_segmenter": {
                            "min_pixel_size": 400,
                            "min_run_time_difference_ms": 1000,
                        },
                        "enabled": True,
                    },
                    "publisher": {
                        "organization_key": "VOXEL_SANDBOX",
                        "enabled": False,
                        "portal_host": "https://app.voxelai.com",
                        # trunk-ignore(gitleaks/generic-api-key)
                        "auth_token": "6bdeb3310546f9cae57c87604f80ece37ba6da43",
                    },
                    "temporal": {
                        "max_frame_segments": 9,
                        "expire_threshold_ms": 120000,
                        "max_future_frames": 10,
                        "max_past_frames": 60,
                    },
                    "state": {
                        "add_epoch_time": False,
                        "frames_to_buffer": 1000,
                        "publisher": {
                            "batch_max_bytes": 1000000,
                            "batch_max_latency_seconds": 60,
                            "batch_max_message": 1000,
                            "retry_deadline_seconds": 600,
                            "state_topic": "projects/local-project/topics/voxel-local-state-messages",
                            "event_topic": "projects/local-project/topics/voxel-local-event-messages",
                            "emulator_host": "127.0.0.1:31002",
                        },
                        "enabled": False,
                    },
                    "video_stream": {"video_source_bucket": "voxel-logs"},
                    "cache_key": "cache_key",
                    "log_key": "",
                    "run_uuid": "uuid",
                    "video_uuid": "americold/modesto/0001/cha/5bb70c8b-b007-476e-a0d8-877d17db336c_0000",
                    "enable_video_writer": False,
                },
            },
            {
                "camera_uuid": "americold/modesto/0001/cha",
                "incidents": [],
                "video_uuid": "americold/modesto/0001/cha/bd282ac3-87bf-4966-98db-43358790131a_0000",
                "config": {
                    "camera": {
                        "arn": "arn:aws:kinesisvideo:us-west-2:360054435465:stream/americold-modesto-0001/1652139618645",
                        "version": 3,
                        "fps": 5,
                        "min_frame_difference_ms": 85,
                    },
                    "camera_uuid": "americold/modesto/0001/cha",
                    "incident": {
                        "should_generate_on_cooldown": True,
                        "incident_machine_params": {
                            "parking": {"max_parked_duration_s": 300}
                        },
                        "monitors_requested": [],
                        "state_machine_monitors_requested": [
                            "bad_posture",
                            "intersection",
                            "parking",
                            "overreaching",
                            "door_intersection",
                            "door_violation",
                            "open_door",
                            "piggyback",
                            "spill",
                        ],
                        "dry_run": True,
                        "temp_directory": "/var/tmp/voxel/incidents",
                        "generate_temp_subdirs": True,
                    },
                    "perception": {
                        "detector_tracker": {
                            "actor2class": {"PERSON": 0, "PIT": 1},
                            "height": 480,
                            "model_path": "artifacts_2021-12-06-00-00-00-0000-yolo/best_480_960.engine",
                            "width": 960,
                        },
                        "door_classifier": {
                            "enabled": True,
                            "model_path": "artifacts_2022-10-26_americold_modesto_0001_cha/2022-10-26_americold_modesto_0001_cha.pt",
                            "model_type": "vanilla_resnet",
                        },
                        "hat_classifier": {"enabled": False},
                        "lift_classifier": {
                            "enabled": True,
                            "model_type": "HM",
                            "model_path": "artifacts_lift_classifier_01282022/lift_classifer_01282022.sav",
                        },
                        "pose": {
                            "enabled": True,
                            "model_path": "artifacts_03_21_2023_pose_0630_jit_update/fast_res50_256x192.pt",
                        },
                        "reach_classifier": {
                            "enabled": True,
                            "model_path": "artifacts_03_23_2023_overreaching_model_jit/voxel_ergo_ml_overreaching_2022-05-23-jit.pt",
                            "model_type": "DL",
                        },
                        "vest_classifier": {"enabled": False},
                        "spill": {
                            "enabled": True,
                            "model_path": "artifacts_02_05_2023_spill_generalized_jit/smp-bd6b62dfd5b64dcfb4970102e2c9b2aa-jit.pt",
                            "min_run_time_difference_ms": 15000,
                            "min_pixel_size": 200,
                            "max_consecutive_runs": 1,
                            "post_process_enabled": True,
                            "frame_segment2class": {"UNKNOWN": 0, "SPILL": 1},
                        },
                        "obstruction_segmenter": {
                            "min_pixel_size": 400,
                            "min_run_time_difference_ms": 1000,
                        },
                        "enabled": True,
                    },
                    "publisher": {
                        "organization_key": "VOXEL_SANDBOX",
                        "enabled": False,
                        "portal_host": "https://app.voxelai.com",
                        # trunk-ignore(gitleaks/generic-api-key)
                        "auth_token": "6bdeb3310546f9cae57c87604f80ece37ba6da43",
                    },
                    "temporal": {
                        "max_frame_segments": 9,
                        "expire_threshold_ms": 120000,
                        "max_future_frames": 10,
                        "max_past_frames": 60,
                    },
                    "state": {
                        "add_epoch_time": False,
                        "frames_to_buffer": 1000,
                        "publisher": {
                            "batch_max_bytes": 1000000,
                            "batch_max_latency_seconds": 60,
                            "batch_max_message": 1000,
                            "retry_deadline_seconds": 600,
                            "state_topic": "projects/local-project/topics/voxel-local-state-messages",
                            "event_topic": "projects/local-project/topics/voxel-local-event-messages",
                            "emulator_host": "127.0.0.1:31002",
                        },
                        "enabled": False,
                    },
                    "video_stream": {"video_source_bucket": "voxel-logs"},
                    "cache_key": "cache_key",
                    "log_key": "",
                    "run_uuid": "uuid",
                    "video_uuid": "americold/modesto/0001/cha/bd282ac3-87bf-4966-98db-43358790131a_0000",
                    "enable_video_writer": False,
                },
            },
        ]

        assert len(graph_configs) == 2

    def test_validate_scale_and_graph_config_incidents(self):
        """
        Test that incidents and scale annotations are properly validated
        """
        self.assertFalse(
            validate_scale_and_graph_config_incidents(
                "americold/modesto/0001/cha",
                {"doors": [{"door_id": 1}]},
                False,
            )
        )

    def test_invalid_door_id(self):
        """
        Test that invalid door ids are properly validated
        """
        self.assertFalse(
            validate_door_id(
                "americold/modesto/0001/cha",
                {"doors": [{"door_id": 0}]},
            )
        )

    def test_gpu_runtime(self):
        config = {"gpu_runtime": {"runtime": "GPU_RUNTIME_BACKEND_LOCAL"}}
        self.assertTrue(
            get_gpu_runtime_from_graph_config(config)
            == GpuRuntimeBackend.GPU_RUNTIME_BACKEND_LOCAL
        )

        config = {
            "gpu_runtime": {"runtime": "GPU_RUNTIME_BACKEND_UNSPECIFIED"}
        }
        self.assertTrue(
            get_gpu_runtime_from_graph_config(config)
            == GpuRuntimeBackend.GPU_RUNTIME_BACKEND_LOCAL
        )

        config = {
            "gpu_runtime": {"runtime": "GPU_RUNTIME_BACKEND_REMOTE_TRITON"}
        }
        self.assertTrue(
            get_gpu_runtime_from_graph_config(config)
            == GpuRuntimeBackend.GPU_RUNTIME_BACKEND_REMOTE_TRITON
        )

        config = {}
        self.assertTrue(
            get_gpu_runtime_from_graph_config(config)
            == GpuRuntimeBackend.GPU_RUNTIME_BACKEND_LOCAL
        )

    def test_get_head_covering_type_from_graph_config(self):
        hard_hat_config = load_yaml_with_jinja(
            "configs/cameras/uscold/quakertown/0001/cha.yaml"
        )
        covered_head_config = load_yaml_with_jinja(
            "configs/cameras/jfe_shoji/burlington/0002/cha.yaml"
        )
        no_hat_config = load_yaml_with_jinja(
            "configs/cameras/americold/modesto/0001/cha.yaml"
        )

        hat_type = get_head_covering_type_from_graph_config(hard_hat_config)
        self.assertEqual(hat_type, HeadCoveringType.HARD_HAT)
        hat_type = get_head_covering_type_from_graph_config(
            covered_head_config
        )
        self.assertEqual(hat_type, HeadCoveringType.COVERED_HEAD)
        hat_type = get_head_covering_type_from_graph_config(no_hat_config)
        self.assertIsNone(hat_type)
