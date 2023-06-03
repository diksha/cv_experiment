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
import unittest

from core.infra.sematic.perception.data_ingestion.ingest_raw_videos.pipeline import (
    VideoIngestionSummary,
)
from core.infra.sematic.perception.regression_scenarios.add_regression_scenario import (
    add_to_regression_set,
)
from core.ml.data.collection.data_collector import Incident


class AddToRegressonScenarioTest(unittest.TestCase):
    def test_add_to_regression_set(self):
        video_ingestion_summary = VideoIngestionSummary(
            scale_task_summary=None,
            ingested_metaverse_videos=["video_uuid_0000"],
            failed_ingested_metaverse_videos=[],
        )
        incidents = [
            Incident(
                camera_uuid="camera_uuid",
                video_s3_path="video_s3_path",
                incident_type_id="incident_type_id",
                incident_uuid="video_uuid",
                organization="organization_id",
                zone="zone_id",
                experimental=False,
                feedback_type="feedback_type",
                scenario_type="scenario_type",
            )
        ]
        self.assertEqual(
            add_to_regression_set(video_ingestion_summary, incidents), None
        )

    def test_add_to_regression_set_raises(self):
        video_ingestion_summary = VideoIngestionSummary(
            scale_task_summary=None,
            ingested_metaverse_videos=["video_uuid"],
            failed_ingested_metaverse_videos=[],
        )
        incidents = [
            Incident(
                camera_uuid="camera_uuid",
                video_s3_path="video_s3_path",
                incident_type_id="incident_type_id",
                incident_uuid="video_uuid",
                organization="organization_id",
                zone="zone_id",
                experimental=False,
                feedback_type="feedback_type",
                scenario_type="scenario_type",
            )
        ]
        with self.assertRaises(RuntimeError):
            add_to_regression_set(video_ingestion_summary, incidents)
        video_ingestion_summary = VideoIngestionSummary(
            scale_task_summary=None,
            ingested_metaverse_videos=["video_uuid_0000"],
            failed_ingested_metaverse_videos=[],
        )
        incidents = []
        with self.assertRaises(KeyError):
            add_to_regression_set(video_ingestion_summary, incidents)
