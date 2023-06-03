#
# Copyright 2020-2022 Voxel Labs, Inc.
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
from typing import Any
from unittest import mock

from scaleapi.tasks import Task

from core.labeling.scale.lib.converters.video_playback_annotation_converter import (
    VideoPlaybackAnnotationConverter,
)
from core.labeling.video_helper_mixin import VideoHelperMixin


# This method will be used by the mock to replace requests.get
def mocked_requests_get(*args, **kwargs) -> Any:
    """Mocks requests.get for scale tasks

    Args:
        kwargs: unused
        args: unused

    Returns:
        Any: response of requests.get
    """

    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self) -> dict:
            """Json of response

            Returns:
                dict: json of response
            """
            return self.json_data

    return MockResponse(
        {
            "d7286ba2-e016-4d32-89e8-cf8bd6bd7454": {
                "label": "PIT_V2",
                "geometry": "box",
                "frames": [
                    {
                        "attributes": {
                            "forklift": "True",
                            "loaded": "False",
                            "human_operating": "True",
                            "forks_raised": "True",
                            "occluded_degree": "Occluded",
                            "truncated": "True",
                        },
                        "left": 826,
                        "top": 115,
                        "height": 104,
                        "width": 93,
                        "key": 0,
                        "timestamp_secs": 0,
                    }
                ],
            },
            "dc63d993-1e05-4190-ae75-6b50460be766": {
                "label": "PERSON_V2",
                "geometry": "box",
                "frames": [
                    {
                        "attributes": {
                            "bend": "Good",
                            "reach": "Good",
                            "lift": "Good",
                            "operating_object": "TRUCK",
                            "occluded_degree": "Occluded",
                            "truncated": "False",
                            "safety_glove1": "43e28463-fa7e-41b6-8b83-304d67434bf0",
                            "head_covered_state": "35f77891-0f2a-426f-94c3-3f0d5a6ced99",
                        },
                        "left": 757,
                        "top": 100,
                        "height": 64,
                        "width": 58,
                        "key": 0,
                        "timestamp_secs": 0,
                    }
                ],
            },
            "dc63d993-1e05-4190-ae75-6b50460be767": {
                "label": "PERSON_V2",
                "geometry": "box",
                "frames": [
                    {
                        "attributes": {
                            "bend": "Good",
                            "reach": "Good",
                            "lift": "Good",
                            "operating_object": "TRUCK",
                            "occluded_degree": "Occluded",
                            "truncated": "False",
                            "safety_glove1": "43e28463-fa7e-41b6-8b83-304d67434bf0",
                            "hard_hat": "43e28463-fa7e-41b6-8b83-304d67434bf1",
                        },
                        "left": 757,
                        "top": 100,
                        "height": 64,
                        "width": 58,
                        "key": 0,
                        "timestamp_secs": 0,
                    }
                ],
            },
            "dc63d993-1e05-4190-ae75-6b50460be768": {
                "label": "PERSON_V2",
                "geometry": "box",
                "frames": [
                    {
                        "attributes": {
                            "bend": "Good",
                            "reach": "Good",
                            "lift": "Good",
                            "operating_object": "TRUCK",
                            "occluded_degree": "Occluded",
                            "truncated": "False",
                            "safety_glove1": "43e28463-fa7e-41b6-8b83-304d67434bf0",
                            "head_covered_state": "35f77891-0f2a-426f-94c3-3f0d5a6cee00",
                            "hard_hat": "43e28463-fa7e-41b6-8b83-304d67434bf1",
                        },
                        "left": 757,
                        "top": 100,
                        "height": 64,
                        "width": 58,
                        "key": 0,
                        "timestamp_secs": 0,
                    }
                ],
            },
            "43e28463-fa7e-41b6-8b83-304d67434bf0": {
                "label": "SAFETY_GLOVE",
                "geometry": "box",
                "frames": [
                    {
                        "attributes": {"occluded_degree": "Occluded"},
                        "left": 728,
                        "top": 224,
                        "height": 67,
                        "width": 99,
                        "key": 0,
                        "timestamp_secs": 0,
                    }
                ],
            },
            "642dd99e-d711-4a20-8205-46b929da4fc8": {
                "label": "TRAILER",
                "geometry": "box",
                "frames": [
                    {
                        "attributes": {
                            "occluded_degree": "HeavilyOccluded",
                            "truncated": "True",
                            "attached_to": "db0cd5b1-ed2f-445a-bb3d-d3ad6eede39f",
                        },
                        "left": 749,
                        "top": 288,
                        "height": 123,
                        "width": 125,
                        "key": 0,
                        "timestamp_secs": 0,
                    }
                ],
            },
            "db0cd5b1-ed2f-445a-bb3d-d3ad6eede39f": {
                "label": "TRUCK",
                "geometry": "box",
                "frames": [
                    {
                        "attributes": {
                            "human_operating": "False",
                            "is_pickup": "True",
                            "occluded_degree": "Occluded",
                            "truncated": "True",
                        },
                        "left": 543,
                        "top": 261,
                        "height": 318,
                        "width": 281,
                        "key": 0,
                        "timestamp_secs": 0,
                    }
                ],
            },
            "35f77891-0f2a-426f-94c3-3f0d5a6ced99": {
                "label": "COVERED_HEAD",
                "geometry": "box",
                "frames": [
                    {
                        "attributes": {
                            "occluded_degree": "Occluded",
                            "truncated": "False",
                        },
                        "left": 757,
                        "top": 361,
                        "height": 10,
                        "width": 93,
                        "key": 0,
                        "timestamp_secs": 0,
                    }
                ],
            },
            "35f77891-0f2a-426f-94c3-3f0d5a6cee00": {
                "label": "COVERED_HEAD",
                "geometry": "box",
                "frames": [
                    {
                        "attributes": {
                            "occluded_degree": "Occluded",
                            "truncated": "False",
                        },
                        "left": 757,
                        "top": 361,
                        "height": 10,
                        "width": 93,
                        "key": 0,
                        "timestamp_secs": 0,
                    }
                ],
            },
        },
        200,
    )


class VideoPlaybackAnnotationConverterTest(unittest.TestCase):
    @mock.patch("requests.get", side_effect=mocked_requests_get)
    @mock.patch.object(
        VideoHelperMixin,
        "get_frame_timestamp_ms_map",
        return_value={0: (0, 720, 720)},
    )
    def test_get_video_from_task(self, mock_frame, mock_get) -> None:
        """Tests getting video from task

        Args:
            mock_frame (mock): mock for frame timestamp
            mock_get (mock): mock for request get annotations
        """
        video_playback_converter = VideoPlaybackAnnotationConverter(
            "2022-12-05",
            "2022-12-01",
            "video_playback_annotation",
            "scale_credentials_arn",
        )
        video = video_playback_converter._get_video_from_task(  # trunk-ignore(pylint/W0212)
            Task(
                json={
                    "task_id": 1,
                    "metadata": {
                        "original_video_path": (
                            "s3://voxel-logs/"
                            "piston_automotive/marion/0002/cha/1.mp4"
                        ),
                        "video_uuid": "piston_automotive/marion/0002/cha/1",
                        "filename": "piston_automotive/marion/0002/cha/1",
                        "taxonomy_version": "123",
                    },
                    "response": {
                        "annotations": {"url": "url"},
                    },
                },
                client=None,
            )
        ).to_dict()
        for frame in video["frames"]:
            for actor in frame["actors"]:
                del actor["uuid"]

        self.assertEqual(
            video,
            {
                "uuid": "piston_automotive/marion/0002/cha/1",
                "parent_uuid": None,
                "root_uuid": None,
                "frames": [
                    {
                        "frame_number": 0,
                        "frame_width": 720,
                        "frame_height": 720,
                        "relative_timestamp_s": 0.0,
                        "relative_timestamp_ms": 0,
                        "epoch_timestamp_ms": None,
                        "frame_segments": None,
                        "actors": [
                            {
                                "category": "PIT_V2",
                                "polygon": {
                                    "vertices": [
                                        {"x": 826, "y": 115, "z": None},
                                        {"x": 919, "y": 115, "z": None},
                                        {"x": 919, "y": 219, "z": None},
                                        {"x": 826, "y": 219, "z": None},
                                    ]
                                },
                                "track_id": None,
                                "track_uuid": "d7286ba2-e016-4d32-89e8-cf8bd6bd7454",
                                "manual": None,
                                "occluded": None,
                                "occluded_degree": "Occluded",
                                "operating_object": None,
                                "truncated": True,
                                "confidence": None,
                                "human_operating": True,
                                "forklift": True,
                                "loaded": False,
                                "forks_raised": True,
                                "operating_pit": None,
                                "door_state": None,
                                "door_state_probabilities": None,
                                "door_type": None,
                                "door_orientation": None,
                                "pose": None,
                                "activity": None,
                                "x_velocity_pixel_per_sec": None,
                                "y_velocity_pixel_per_sec": None,
                                "x_velocity_meters_per_sec": None,
                                "y_velocity_meters_per_sec": None,
                                "x_position_m": None,
                                "y_position_m": None,
                                "z_position_m": None,
                                "distance_to_camera_m": None,
                                "is_wearing_safety_vest": None,
                                "is_pickup": None,
                                "is_motorized": None,
                                "is_van": None,
                                "is_wearing_safety_vest_v2": None,
                                "is_wearing_hard_hat": None,
                                "motion_detection_zone_state": None,
                                "is_carrying_object": None,
                                "motion_detection_score_std": None,
                                "head_covering_type": None,
                                "skeleton": None,
                                "ergonomic_severity_metrics": None,
                            },
                            {
                                "category": "PERSON_V2",
                                "polygon": {
                                    "vertices": [
                                        {"x": 757, "y": 100, "z": None},
                                        {"x": 815, "y": 100, "z": None},
                                        {"x": 815, "y": 164, "z": None},
                                        {"x": 757, "y": 164, "z": None},
                                    ]
                                },
                                "track_id": None,
                                "track_uuid": "dc63d993-1e05-4190-ae75-6b50460be766",
                                "manual": None,
                                "occluded": None,
                                "occluded_degree": "Occluded",
                                "operating_object": "TRUCK",
                                "truncated": False,
                                "confidence": None,
                                "human_operating": None,
                                "forklift": None,
                                "loaded": None,
                                "forks_raised": None,
                                "operating_pit": None,
                                "door_state": None,
                                "door_state_probabilities": None,
                                "door_type": None,
                                "door_orientation": None,
                                "pose": None,
                                "activity": {
                                    "LIFTING": "GOOD",
                                    "REACHING": "GOOD",
                                },
                                "x_velocity_pixel_per_sec": None,
                                "y_velocity_pixel_per_sec": None,
                                "x_velocity_meters_per_sec": None,
                                "y_velocity_meters_per_sec": None,
                                "x_position_m": None,
                                "y_position_m": None,
                                "z_position_m": None,
                                "distance_to_camera_m": None,
                                "is_wearing_safety_vest": None,
                                "is_pickup": None,
                                "is_motorized": None,
                                "is_van": None,
                                "is_wearing_safety_vest_v2": None,
                                "is_wearing_hard_hat": False,
                                "motion_detection_zone_state": None,
                                "is_carrying_object": None,
                                "motion_detection_score_std": None,
                                "head_covering_type": "COVERED_HEAD",
                                "skeleton": None,
                                "ergonomic_severity_metrics": None,
                            },
                            {
                                "category": "PERSON_V2",
                                "polygon": {
                                    "vertices": [
                                        {"x": 757, "y": 100, "z": None},
                                        {"x": 815, "y": 100, "z": None},
                                        {"x": 815, "y": 164, "z": None},
                                        {"x": 757, "y": 164, "z": None},
                                    ]
                                },
                                "track_id": None,
                                "track_uuid": "dc63d993-1e05-4190-ae75-6b50460be767",
                                "manual": None,
                                "occluded": None,
                                "occluded_degree": "Occluded",
                                "operating_object": "TRUCK",
                                "truncated": False,
                                "confidence": None,
                                "human_operating": None,
                                "forklift": None,
                                "loaded": None,
                                "forks_raised": None,
                                "operating_pit": None,
                                "door_state": None,
                                "door_state_probabilities": None,
                                "door_type": None,
                                "door_orientation": None,
                                "pose": None,
                                "activity": {
                                    "LIFTING": "GOOD",
                                    "REACHING": "GOOD",
                                },
                                "x_velocity_pixel_per_sec": None,
                                "y_velocity_pixel_per_sec": None,
                                "x_velocity_meters_per_sec": None,
                                "y_velocity_meters_per_sec": None,
                                "x_position_m": None,
                                "y_position_m": None,
                                "z_position_m": None,
                                "distance_to_camera_m": None,
                                "is_wearing_safety_vest": None,
                                "is_pickup": None,
                                "is_motorized": None,
                                "is_van": None,
                                "is_wearing_safety_vest_v2": None,
                                "is_wearing_hard_hat": True,
                                "motion_detection_zone_state": None,
                                "is_carrying_object": None,
                                "motion_detection_score_std": None,
                                "head_covering_type": "HARD_HAT",
                                "skeleton": None,
                                "ergonomic_severity_metrics": None,
                            },
                            {
                                "category": "PERSON_V2",
                                "polygon": {
                                    "vertices": [
                                        {"x": 757, "y": 100, "z": None},
                                        {"x": 815, "y": 100, "z": None},
                                        {"x": 815, "y": 164, "z": None},
                                        {"x": 757, "y": 164, "z": None},
                                    ]
                                },
                                "track_id": None,
                                "track_uuid": "dc63d993-1e05-4190-ae75-6b50460be768",
                                "manual": None,
                                "occluded": None,
                                "occluded_degree": "Occluded",
                                "operating_object": "TRUCK",
                                "truncated": False,
                                "confidence": None,
                                "human_operating": None,
                                "forklift": None,
                                "loaded": None,
                                "forks_raised": None,
                                "operating_pit": None,
                                "door_state": None,
                                "door_state_probabilities": None,
                                "door_type": None,
                                "door_orientation": None,
                                "pose": None,
                                "activity": {
                                    "LIFTING": "GOOD",
                                    "REACHING": "GOOD",
                                },
                                "x_velocity_pixel_per_sec": None,
                                "y_velocity_pixel_per_sec": None,
                                "x_velocity_meters_per_sec": None,
                                "y_velocity_meters_per_sec": None,
                                "x_position_m": None,
                                "y_position_m": None,
                                "z_position_m": None,
                                "distance_to_camera_m": None,
                                "is_wearing_safety_vest": None,
                                "is_pickup": None,
                                "is_motorized": None,
                                "is_van": None,
                                "is_wearing_safety_vest_v2": None,
                                "is_wearing_hard_hat": False,
                                "motion_detection_zone_state": None,
                                "is_carrying_object": None,
                                "motion_detection_score_std": None,
                                "head_covering_type": "COVERED_HEAD",
                                "skeleton": None,
                                "ergonomic_severity_metrics": None,
                            },
                            {
                                "category": "SAFETY_GLOVE",
                                "polygon": {
                                    "vertices": [
                                        {"x": 728, "y": 224, "z": None},
                                        {"x": 827, "y": 224, "z": None},
                                        {"x": 827, "y": 291, "z": None},
                                        {"x": 728, "y": 291, "z": None},
                                    ]
                                },
                                "track_id": None,
                                "track_uuid": "43e28463-fa7e-41b6-8b83-304d67434bf0",
                                "manual": None,
                                "occluded": None,
                                "occluded_degree": "Occluded",
                                "operating_object": None,
                                "truncated": None,
                                "confidence": None,
                                "human_operating": None,
                                "forklift": None,
                                "loaded": None,
                                "forks_raised": None,
                                "operating_pit": None,
                                "door_state": None,
                                "door_state_probabilities": None,
                                "door_type": None,
                                "door_orientation": None,
                                "pose": None,
                                "activity": None,
                                "x_velocity_pixel_per_sec": None,
                                "y_velocity_pixel_per_sec": None,
                                "x_velocity_meters_per_sec": None,
                                "y_velocity_meters_per_sec": None,
                                "x_position_m": None,
                                "y_position_m": None,
                                "z_position_m": None,
                                "distance_to_camera_m": None,
                                "is_wearing_safety_vest": None,
                                "is_pickup": None,
                                "is_motorized": None,
                                "is_van": None,
                                "is_wearing_safety_vest_v2": None,
                                "is_wearing_hard_hat": None,
                                "motion_detection_zone_state": None,
                                "is_carrying_object": None,
                                "motion_detection_score_std": None,
                                "head_covering_type": None,
                                "skeleton": None,
                                "ergonomic_severity_metrics": None,
                            },
                            {
                                "category": "TRAILER",
                                "polygon": {
                                    "vertices": [
                                        {"x": 749, "y": 288, "z": None},
                                        {"x": 874, "y": 288, "z": None},
                                        {"x": 874, "y": 411, "z": None},
                                        {"x": 749, "y": 411, "z": None},
                                    ]
                                },
                                "track_id": None,
                                "track_uuid": "642dd99e-d711-4a20-8205-46b929da4fc8",
                                "manual": None,
                                "occluded": None,
                                "occluded_degree": "HeavilyOccluded",
                                "operating_object": None,
                                "truncated": True,
                                "confidence": None,
                                "human_operating": None,
                                "forklift": None,
                                "loaded": None,
                                "forks_raised": None,
                                "operating_pit": None,
                                "door_state": None,
                                "door_state_probabilities": None,
                                "door_type": None,
                                "door_orientation": None,
                                "pose": None,
                                "activity": None,
                                "x_velocity_pixel_per_sec": None,
                                "y_velocity_pixel_per_sec": None,
                                "x_velocity_meters_per_sec": None,
                                "y_velocity_meters_per_sec": None,
                                "x_position_m": None,
                                "y_position_m": None,
                                "z_position_m": None,
                                "distance_to_camera_m": None,
                                "is_wearing_safety_vest": None,
                                "is_pickup": None,
                                "is_motorized": None,
                                "is_van": None,
                                "is_wearing_safety_vest_v2": None,
                                "is_wearing_hard_hat": None,
                                "motion_detection_zone_state": None,
                                "is_carrying_object": None,
                                "motion_detection_score_std": None,
                                "head_covering_type": None,
                                "skeleton": None,
                                "ergonomic_severity_metrics": None,
                            },
                            {
                                "category": "TRUCK",
                                "polygon": {
                                    "vertices": [
                                        {"x": 543, "y": 261, "z": None},
                                        {"x": 824, "y": 261, "z": None},
                                        {"x": 824, "y": 579, "z": None},
                                        {"x": 543, "y": 579, "z": None},
                                    ]
                                },
                                "track_id": None,
                                "track_uuid": "db0cd5b1-ed2f-445a-bb3d-d3ad6eede39f",
                                "manual": None,
                                "occluded": None,
                                "occluded_degree": "Occluded",
                                "operating_object": None,
                                "truncated": True,
                                "confidence": None,
                                "human_operating": False,
                                "forklift": None,
                                "loaded": None,
                                "forks_raised": None,
                                "operating_pit": None,
                                "door_state": None,
                                "door_state_probabilities": None,
                                "door_type": None,
                                "door_orientation": None,
                                "pose": None,
                                "activity": None,
                                "x_velocity_pixel_per_sec": None,
                                "y_velocity_pixel_per_sec": None,
                                "x_velocity_meters_per_sec": None,
                                "y_velocity_meters_per_sec": None,
                                "x_position_m": None,
                                "y_position_m": None,
                                "z_position_m": None,
                                "distance_to_camera_m": None,
                                "is_wearing_safety_vest": None,
                                "is_pickup": True,
                                "is_motorized": None,
                                "is_van": None,
                                "is_wearing_safety_vest_v2": None,
                                "is_wearing_hard_hat": None,
                                "motion_detection_zone_state": None,
                                "is_carrying_object": None,
                                "motion_detection_score_std": None,
                                "head_covering_type": None,
                                "skeleton": None,
                                "ergonomic_severity_metrics": None,
                            },
                            {
                                "category": "COVERED_HEAD",
                                "polygon": {
                                    "vertices": [
                                        {"x": 757, "y": 361, "z": None},
                                        {"x": 850, "y": 361, "z": None},
                                        {"x": 850, "y": 371, "z": None},
                                        {"x": 757, "y": 371, "z": None},
                                    ]
                                },
                                "track_id": None,
                                "track_uuid": "35f77891-0f2a-426f-94c3-3f0d5a6ced99",
                                "manual": None,
                                "occluded": None,
                                "occluded_degree": "Occluded",
                                "operating_object": None,
                                "truncated": False,
                                "confidence": None,
                                "human_operating": None,
                                "forklift": None,
                                "loaded": None,
                                "forks_raised": None,
                                "operating_pit": None,
                                "door_state": None,
                                "door_state_probabilities": None,
                                "door_type": None,
                                "door_orientation": None,
                                "pose": None,
                                "activity": None,
                                "x_velocity_pixel_per_sec": None,
                                "y_velocity_pixel_per_sec": None,
                                "x_velocity_meters_per_sec": None,
                                "y_velocity_meters_per_sec": None,
                                "x_position_m": None,
                                "y_position_m": None,
                                "z_position_m": None,
                                "distance_to_camera_m": None,
                                "is_wearing_safety_vest": None,
                                "is_pickup": None,
                                "is_motorized": None,
                                "is_van": None,
                                "is_wearing_safety_vest_v2": None,
                                "is_wearing_hard_hat": None,
                                "motion_detection_zone_state": None,
                                "is_carrying_object": None,
                                "motion_detection_score_std": None,
                                "head_covering_type": None,
                                "skeleton": None,
                                "ergonomic_severity_metrics": None,
                            },
                            {
                                "category": "COVERED_HEAD",
                                "polygon": {
                                    "vertices": [
                                        {"x": 757, "y": 361, "z": None},
                                        {"x": 850, "y": 361, "z": None},
                                        {"x": 850, "y": 371, "z": None},
                                        {"x": 757, "y": 371, "z": None},
                                    ]
                                },
                                "track_id": None,
                                "track_uuid": "35f77891-0f2a-426f-94c3-3f0d5a6cee00",
                                "manual": None,
                                "occluded": None,
                                "occluded_degree": "Occluded",
                                "operating_object": None,
                                "truncated": False,
                                "confidence": None,
                                "human_operating": None,
                                "forklift": None,
                                "loaded": None,
                                "forks_raised": None,
                                "operating_pit": None,
                                "door_state": None,
                                "door_state_probabilities": None,
                                "door_type": None,
                                "door_orientation": None,
                                "pose": None,
                                "activity": None,
                                "x_velocity_pixel_per_sec": None,
                                "y_velocity_pixel_per_sec": None,
                                "x_velocity_meters_per_sec": None,
                                "y_velocity_meters_per_sec": None,
                                "x_position_m": None,
                                "y_position_m": None,
                                "z_position_m": None,
                                "distance_to_camera_m": None,
                                "is_wearing_safety_vest": None,
                                "is_pickup": None,
                                "is_motorized": None,
                                "is_van": None,
                                "is_wearing_safety_vest_v2": None,
                                "is_wearing_hard_hat": None,
                                "motion_detection_zone_state": None,
                                "is_carrying_object": None,
                                "motion_detection_score_std": None,
                                "head_covering_type": None,
                                "skeleton": None,
                                "ergonomic_severity_metrics": None,
                            },
                        ],
                        "relative_image_path": None,
                    }
                ],
                "voxel_uuid": None,
                "is_test": None,
            },
        )
