import unittest
from unittest.mock import patch

from core.execution.utils.graph_config_builder import GraphConfigBuilder

# trunk-ignore-all(pylint/C0301): line too long


class GraphConfigBuilderTest(unittest.TestCase):
    @patch("core.execution.utils.graph_config_builder.logger")
    def test_sample_config(self, mock_logger):
        expected_merged_config = {
            "camera": {
                "arn": "arn:aws:kinesisvideo:us-west-2:360054435465:stream/wesco-reno-0002/1673393133764",
                "fps": 5,
                "version": 1,
            },
            "camera_uuid": "wesco/reno/0002/cha",
            "incident": {
                "dry_run": False,
                "generate_temp_subdirs": True,
                "incident_machine_params": {
                    "parking": {"max_parked_duration_s": 300},
                    "safety_vest": {"per_camera_cooldown_s": 3600},
                },
                "monitors_requested": [],
                "state_machine_monitors_requested": [
                    "bad_posture",
                    "overreaching",
                    "parking",
                    "safety_vest",
                ],
                "temp_directory": "/var/tmp/voxel/incidents",
            },
            "perception": {
                "detector_tracker": {
                    "actor2class": {"PERSON": 0, "PIT": 1},
                    "height": 736,
                    "model_path": "artifacts_01_18_2023_piston-automotive-yolo/best_736_1280.engine",
                    "width": 1280,
                },
                "door_classifier": {"enabled": False},
                "hat_classifier": {"enabled": False},
                "lift_classifier": {
                    "enabled": True,
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
                "vest_classifier": {
                    "enabled": True,
                    "min_actor_pixel_area": 2000,
                    "model_path": "artifacts_voxel_safetyvest_vit_general_2022-09-21/voxel_safetyvest_vit_general_2022-09-21.pth",
                    "model_type": "Transformers",
                },
            },
            "publisher": {
                "auth_token": "sample_auth_token",
                "enabled": True,
                "organization_key": "WESCO",
                "portal_host": "https://app.voxelai.com",
            },
            "state": {
                "add_epoch_time": False,
                "enabled": True,
                "frames_to_buffer": 1000,
                "publisher": {
                    "batch_max_bytes": 1000000,
                    "batch_max_latency_seconds": 60,
                    "batch_max_message": 1000,
                    "emulator_host": None,
                    "event_topic": "projects/sodium-carving-227300/topics/voxel-production-event-messages",
                    "retry_deadline_seconds": 600,
                    "state_topic": "projects/sodium-carving-227300/topics/voxel-production-state-messages",
                },
            },
            "temporal": {
                "expire_threshold_ms": 120000,
                "max_future_frames": 30,
                "max_past_frames": 60,
            },
            "run_uuid": ":4e727983-17a1-40aa-87e9-025ddd7938ed",
        }

        default_config = {
            "camera": {"fps": 5},
            "state": {
                "add_epoch_time": False,
                "frames_to_buffer": 1000,
                "publisher": {
                    "batch_max_bytes": 1000000,
                    "batch_max_latency_seconds": 60,
                    "batch_max_message": 1000,
                    "retry_deadline_seconds": 600,
                },
            },
            "temporal": {
                "expire_threshold_ms": 120000,
                "max_future_frames": 30,
                "max_past_frames": 60,
            },
        }
        camera_config = {
            "camera": {
                "arn": "arn:aws:kinesisvideo:us-west-2:360054435465:stream/wesco-reno-0002/1673393133764",
                "fps": 5,
                "version": 1,
            },
            "camera_uuid": "wesco/reno/0002/cha",
            "incident": {
                "dry_run": False,
                "generate_temp_subdirs": True,
                "incident_machine_params": {
                    "parking": {"max_parked_duration_s": 300},
                    "safety_vest": {"per_camera_cooldown_s": 3600},
                },
                "monitors_requested": [],
                "state_machine_monitors_requested": [
                    "bad_posture",
                    "overreaching",
                    "parking",
                    "safety_vest",
                ],
                "temp_directory": "/var/tmp/voxel/incidents",
            },
            "perception": {
                "detector_tracker": {
                    "actor2class": {"PERSON": 0, "PIT": 1},
                    "height": 736,
                    "model_path": "artifacts_01_18_2023_piston-automotive-yolo/best_736_1280.engine",
                    "width": 1280,
                },
                "door_classifier": {"enabled": False},
                "hat_classifier": {"enabled": False},
                "lift_classifier": {
                    "enabled": True,
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
                "vest_classifier": {
                    "enabled": True,
                    "min_actor_pixel_area": 2000,
                    "model_path": "artifacts_voxel_safetyvest_vit_general_2022-09-21/voxel_safetyvest_vit_general_2022-09-21.pth",
                    "model_type": "Transformers",
                },
            },
            "publisher": {
                "auth_token": "sample_auth_token",
                "enabled": True,
                "organization_key": "WESCO",
                "portal_host": "https://app.voxelai.com",
            },
            "state": {
                "add_epoch_time": False,
                "enabled": True,
                "frames_to_buffer": 1000,
                "publisher": {
                    "batch_max_bytes": 1000000,
                    "batch_max_latency_seconds": 60,
                    "batch_max_message": 1000,
                    "emulator_host": None,
                    "event_topic": "projects/sodium-carving-227300/topics/voxel-production-event-messages",
                    "retry_deadline_seconds": 600,
                    "state_topic": "projects/sodium-carving-227300/topics/voxel-production-state-messages",
                },
            },
            "temporal": {
                "expire_threshold_ms": 120000,
                "max_future_frames": 30,
                "max_past_frames": 60,
            },
        }
        env_config = {
            "incident": {"dry_run": False},
            "publisher": {
                "enabled": True,
                "portal_host": "https://app.voxelai.com",
            },
            "state": {
                "enabled": True,
                "publisher": {
                    "emulator_host": None,
                    "event_topic": "projects/sodium-carving-227300/topics/voxel-production-event-messages",
                    "state_topic": "projects/sodium-carving-227300/topics/voxel-production-state-messages",
                },
            },
        }

        # Generated value
        del expected_merged_config["run_uuid"]

        builder = GraphConfigBuilder()
        builder.apply(default_config, "default")
        builder.apply(camera_config, "camera")
        builder.apply(env_config, "environment")

        merged_config = builder.get_config()

        assert mock_logger.debug.call_count == 1
        assert mock_logger.debug.call_args[0][0].startswith(
            "camera.fps set in camera - overrides default"
        )
        assert expected_merged_config == merged_config
