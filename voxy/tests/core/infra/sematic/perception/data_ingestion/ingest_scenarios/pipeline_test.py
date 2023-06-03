#
# Copyright 2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import os

# lightly hack to avoid running network call upon import
# data_collector imports via core/metaverse/api/queries.py
# https://github.com/lightly-ai/lightly/blob/fc252424afae73af54826d97b36042130522d025/lightly/__init__.py#L126
# trunk-ignore-all(pylint/C0413)
os.environ["LIGHTLY_DID_VERSION_CHECK"] = "True"
import unittest

from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.perception.data_ingestion.ingest_raw_videos.pipeline import (
    VideoIngestionSummary,
)
from core.infra.sematic.perception.data_ingestion.ingest_raw_videos.pipeline import (
    pipeline as raw_video_ingestion_pipeline,
)
from core.infra.sematic.perception.data_ingestion.ingest_scenarios.pipeline import (
    copy_incidents_to_voxel_raw_logs,
    generate_scenario_config,
    pipeline,
)
from core.infra.sematic.shared.mock_sematic_funcs import mock_sematic_funcs
from core.infra.sematic.shared.utils import PipelineSetup
from core.labeling.scale.runners.create_scale_tasks import ScaleTaskSummary
from core.ml.data.collection.data_collector import (
    Incident,
    select_incidents_from_portal,
)
from core.structs.data_collection import DataCollectionType


class PipelineTest(unittest.TestCase):
    """Tests for the video ingestion pipeline"""

    def test_setup_connections(self):
        """Confirms pipeline produces no errors and the ingestion results"""
        pipeline_funcs = [
            select_incidents_from_portal,
            copy_incidents_to_voxel_raw_logs,
            raw_video_ingestion_pipeline,
            generate_scenario_config,
        ]
        with mock_sematic_funcs(funcs=pipeline_funcs) as func_mocks:
            func_mocks[select_incidents_from_portal].mock.return_value = [
                Incident(
                    camera_uuid="americold/modesto/0001/cha",
                    video_s3_path=(
                        "s3://fake-s3-bucket/americold/modesto/incidents/uuid1_video.mp4"
                    ),
                    incident_type_id="SAFETY_VEST",
                    incident_uuid="uuid2",
                    organization="americold",
                    zone="modesto",
                    experimental=True,
                    feedback_type="valid",
                    scenario_type="POSITIVE",
                ),
                Incident(
                    camera_uuid="americold/ontario/0001/cha",
                    video_s3_path=(
                        "s3://fake-s3-bucket/americold/ontario/incidents/uuid2_video.mp4"
                    ),
                    incident_type_id="SAFETY_VEST",
                    incident_uuid="uuid2",
                    organization="americold",
                    zone="ontario",
                    experimental=True,
                    feedback_type="valid",
                    scenario_type="POSITIVE",
                ),
                Incident(
                    camera_uuid="americold/modesto/0003/cha",
                    video_s3_path=(
                        "s3://fake-s3-bucket/americold/modesto/incidents/uuid3_video.mp4"
                    ),
                    incident_type_id="SAFETY_VEST",
                    incident_uuid="uuid3",
                    organization="americold",
                    zone="modesto",
                    experimental=True,
                    feedback_type="invalid",
                    scenario_type="NEGATIVE",
                ),
                Incident(
                    camera_uuid="americold/ontario/0003/cha",
                    video_s3_path=(
                        "s3://fake-s3-bucket/americold/ontario/incidents/uuid4_video.mp4"
                    ),
                    incident_type_id="SAFETY_VEST",
                    incident_uuid="uuid4",
                    organization="americold",
                    zone="ontario",
                    experimental=True,
                    feedback_type="invalid",
                    scenario_type="NEGATIVE",
                ),
            ]
            func_mocks[copy_incidents_to_voxel_raw_logs].mock.return_value = [
                "americold/modesto/0001/cha/scenarios/SAFETY_VEST/POSITIVE/uuid1",
                "americold/ontario/0001/cha/scenarios/SAFETY_VEST/POSITIVE/uuid2",
                "americold/modesto/0003/cha/scenarios/SAFETY_VEST/NEGATIVE/uuid3",
                "americold/ontario/0003/cha/scenarios/SAFETY_VEST/NEGATIVE/uuid4",
            ]
            func_mocks[
                raw_video_ingestion_pipeline
            ].mock.return_value = VideoIngestionSummary(
                scale_task_summary=ScaleTaskSummary(
                    video_uuids=[
                        "americold/modesto/0001/cha/scenarios/SAFETY_VEST/POSITIVE/uuid1_0000",
                        "americold/ontario/0001/cha/scenarios/SAFETY_VEST/POSITIVE/uuid2_0000",
                        "americold/modesto/0003/cha/scenarios/SAFETY_VEST/NEGATIVE/uuid3_0000",
                        "americold/ontario/0003/cha/scenarios/SAFETY_VEST/NEGATIVE/uuid4_0000",
                    ],
                    fps=0,
                    task_type="SafetyVestImageAnnotationTask",
                    data_type=DataCollectionType.IMAGE_COLLECTION,
                    batch_name="batch_23_01_2023_21_34",
                    task_unique_ids=[
                        "task_uuid1",
                        "task_uuid2",
                        "task_uuid3",
                        "task_uuid4",
                    ],
                    generate_hypothesis=False,
                ),
                ingested_metaverse_videos=[
                    "americold/modesto/0001/cha/scenarios/SAFETY_VEST/POSITIVE/uuid1_0000",
                    "americold/ontario/0001/cha/scenarios/SAFETY_VEST/POSITIVE/uuid2_0000",
                    "americold/modesto/0003/cha/scenarios/SAFETY_VEST/NEGATIVE/uuid3_0000",
                    "americold/ontario/0003/cha/scenarios/SAFETY_VEST/NEGATIVE/uuid4_0000",
                ],
                failed_ingested_metaverse_videos=[],
            )
            func_mocks[
                generate_scenario_config
            ].mock.return_value = (
                "s3://voxel-temp/ingest_scenarios/run_uuid1/scenario_set.yaml"
            )
            summary: VideoIngestionSummary
            s3_path: str
            # trunk-ignore(pylint/E1101)
            summary, s3_path = pipeline(
                incident_type="safety_vest",
                camera_uuids=[
                    "americold/modesto/0001/cha",
                    "americold/ontario/0001/cha",
                    "americold/modesto/0003/cha",
                    "americold/ontario/0003/cha",
                ],
                start_date=None,
                end_date=None,
                max_incidents=2,
                environment="dev",
                experimental_incidents_only=True,
                is_test=True,
                fps=0.0,
                metaverse_environment="INTERNAL",
                pipeline_setup=PipelineSetup(),
            ).resolve(SilentResolver())
            self.assertIsInstance(summary, VideoIngestionSummary)
            self.assertIsInstance(s3_path, str)
