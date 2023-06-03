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

import unittest

import mock
from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.perception.data_ingestion.ingest_raw_videos.pipeline import (
    VideoIngestionSummary,
    generate_video_ingest_summary,
    pipeline,
)
from core.infra.sematic.shared.mock_sematic_funcs import mock_sematic_funcs
from core.infra.sematic.shared.utils import PipelineSetup
from core.labeling.logs_store.chunk_and_upload_video_logs import (
    VideoIngestInput,
    chunk_and_upload_video_logs,
    create_video_ingest_input,
)
from core.labeling.logs_store.ingest_data_collection_metaverse import (
    DataCollectionInfo,
    generate_data_collection_metadata_list,
    ingest_data_collections_to_metaverse,
)
from core.labeling.scale.runners.create_scale_tasks import (
    ScaleTaskSummary,
    create_scale_tasks,
)
from core.structs.data_collection import DataCollectionType


class PipelineTest(unittest.TestCase):
    """Tests for the video ingestion pipeline"""

    def setUp(self) -> None:
        """Setup ivars for testing"""
        self.input_video_uuid = (
            "ppg/cedar_falls/0011/cha/scenarios/BUMP_CAP/negative"
            "/bd682993-d242-4042-a76e-135b9fce697f"
        )
        self.chunked_video_uuid = (
            "ppg/cedar_falls/0011/cha/scenarios/BUMP_CAP/negative/"
            "bd682993-d242-4042-a76e-135b9fce697f_0000"
        )
        self.scale_task_summary = ScaleTaskSummary(
            video_uuids=[self.chunked_video_uuid],
            fps=0,
            task_type="VideoPlaybackAnnotationTask",
            data_type=DataCollectionType.VIDEO,
            batch_name="my_batch_name",
            task_unique_ids=["scale_uuid_1"],
            generate_hypothesis=False,
        )

    @mock.patch(
        "core.infra.sematic.perception.data_ingestion.ingest_raw_videos."
        "pipeline.is_data_collection_in_metaverse"
    )
    def test_setup_connections(self, is_data_collection_in_metaverse):
        """Confirms pipeline produces no errors and the ingestion results"""
        is_data_collection_in_metaverse.return_value = False
        pipeline_funcs = [
            create_video_ingest_input,
            chunk_and_upload_video_logs,
            create_scale_tasks,
            generate_data_collection_metadata_list,
            ingest_data_collections_to_metaverse,
            generate_video_ingest_summary,
        ]
        with mock_sematic_funcs(funcs=pipeline_funcs) as func_mocks:
            func_mocks[
                create_video_ingest_input
            ].mock.return_value = VideoIngestInput(
                video_uuids=[self.input_video_uuid]
            )
            func_mocks[chunk_and_upload_video_logs].mock.return_value = [
                self.chunked_video_uuid,
                [
                    DataCollectionInfo(
                        data_collection_uuid=self.input_video_uuid,
                        is_test=False,
                        data_collection_type=DataCollectionType.VIDEO,
                    )
                ],
            ]
            func_mocks[
                create_scale_tasks
            ].mock.return_value = self.scale_task_summary
            func_mocks[
                generate_data_collection_metadata_list
            ].mock.return_value = [
                DataCollectionInfo(
                    data_collection_uuid=self.chunked_video_uuid,
                    is_test=False,
                    data_collection_type=DataCollectionType.VIDEO,
                )
            ]
            func_mocks[
                ingest_data_collections_to_metaverse
            ].mock.return_value = ([self.chunked_video_uuid], [])
            func_mocks[
                generate_video_ingest_summary
            ].mock.return_value = VideoIngestionSummary(
                scale_task_summary=self.scale_task_summary,
                ingested_metaverse_videos=self.chunked_video_uuid,
                failed_ingested_metaverse_videos=[],
            )
            # trunk-ignore(pylint/E1101)
            result: VideoIngestionSummary = pipeline(
                videos=self.input_video_uuid,
                is_test=False,
                fps=0.0,
                metaverse_environment="INTERNAL",
                pipeline_setup=PipelineSetup(),
            ).resolve(SilentResolver())
            self.assertIsInstance(result, VideoIngestionSummary)
