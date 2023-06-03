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
import os

# lightly hack to avoid running network call upon import
# https://github.com/lightly-ai/lightly/blob/fc252424afae73af54826d97b36042130522d025/lightly/__init__.py#L126
# trunk-ignore-all(pylint/C0413,flake8/E402)
os.environ["LIGHTLY_DID_VERSION_CHECK"] = "True"
import unittest
from unittest.mock import Mock, patch

from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.perception.data_ingestion.door_dataflywheel.pipeline import (
    IngestionSummary,
    door_dataflywheel,
)
from core.infra.sematic.shared.mock_sematic_funcs import mock_sematic_funcs
from core.infra.sematic.shared.utils import PipelineSetup
from core.labeling.logs_store.chunk_and_upload_video_logs import (
    chunk_and_upload_video_logs,
)
from core.labeling.logs_store.ingest_data_collection_metaverse import (
    ingest_data_collections_to_metaverse,
)
from core.labeling.scale.runners.create_scale_tasks import (
    ScaleTaskSummary,
    create_scale_tasks,
)
from core.ml.data.collection.data_collector import (
    Incident,
    select_incidents_from_portal,
)
from core.ml.data.curation.crop_s3_videos import crop_all_doors_from_videos
from core.ml.data.curation.lib.lightly_worker import (
    LightlyVideoFrameSequence,
    run_lightly_worker,
)
from core.ml.data.curation.prepare_lightly_run import prepare_lightly_run
from core.ml.data.curation.trim_lightly_clips import (
    DataCollectionInfo,
    TrimmedVideoSummary,
    trim_lightly_clips,
)
from core.structs.data_collection import DataCollectionType
from core.structs.task import Task


class PipelineTest(unittest.TestCase):
    """Tests for the Door Data ingestion pipeline"""

    @patch(
        (
            "core.infra.sematic.perception.data_ingestion."
            "door_dataflywheel.pipeline.get_or_create_task_and_service"
        )
    )
    def test_step_connections(
        self,
        mock_task_query: Mock,
    ):
        """Confirms pipeline produces no errors and the flywheel summary
        Args:
            mock_task_query (Mock): task query mock
        """
        mock_task_query.return_value = Task(
            purpose="DOOR_STATE",
            camera_uuids=["americold/modesto/0011/cha"],
        )
        pipeline_funcs = [
            select_incidents_from_portal,
            crop_all_doors_from_videos,
            prepare_lightly_run,
            trim_lightly_clips,
            chunk_and_upload_video_logs,
            create_scale_tasks,
            ingest_data_collections_to_metaverse,
            run_lightly_worker,
        ]
        with mock_sematic_funcs(funcs=pipeline_funcs) as func_mocks:
            func_mocks[select_incidents_from_portal].mock.return_value = [
                Incident(
                    camera_uuid="americold/ontario/0005/cha",
                    video_s3_path="s3://path1",
                    incident_type_id="incident",
                    incident_uuid="uuid1",
                    organization="americold",
                    zone="ontario",
                    experimental=True,
                    feedback_type="",
                    scenario_type="",
                    copied_s3_path="s3://path2",
                ),
            ]
            func_mocks[
                crop_all_doors_from_videos
            ].mock.return_value = (
                "s3://cropped-videos-bucket/cropped/videos/path"
            )
            func_mocks[prepare_lightly_run].mock.return_value = dict(
                dataset_name="lightly-dataset-name",
                dataset_id="lightly-dataset-id",
                input_bucket="lightly-input-bucket",
                output_bucket="lightly-output-bucket",
                input_dir="cropped/videos/path",
                output_dir="lightly/videos/path",
            )
            func_mocks[run_lightly_worker].mock.return_value = [
                LightlyVideoFrameSequence(
                    video_name="video_name1",
                    frame_names=["frame1_name.png", "frame2_name.png"],
                    frame_indices=[1, 2, 3],
                    frame_timestamps_pts=[10, 20, 30],
                    frame_timestamps_sec=[1, 2, 3],
                ),
                LightlyVideoFrameSequence(
                    video_name="video_name2",
                    frame_names=["frame1_name.png", "frame2_name.png"],
                    frame_indices=[1, 2, 3],
                    frame_timestamps_pts=[10, 20, 30],
                    frame_timestamps_sec=[1, 2, 3],
                ),
            ]
            videos_to_ingest = [
                DataCollectionInfo(
                    data_collection_uuid="to_ingest_video_id_1",
                    is_test=True,
                    data_collection_type=DataCollectionType.VIDEO,
                ),
                DataCollectionInfo(
                    data_collection_uuid="to_ingest_video_id_2",
                    is_test=True,
                    data_collection_type=DataCollectionType.VIDEO,
                ),
            ]
            func_mocks[
                trim_lightly_clips
            ].mock.return_value = TrimmedVideoSummary(
                output_bucket="trimmed-videos-bucket",
                video_uuids=["video-uuid-1", "video-uuid-2"],
                failed_video_names=["failed-video-1", "failed-video-2"],
                to_ingest_videos=videos_to_ingest,
            )
            func_mocks[chunk_and_upload_video_logs].mock.return_value = (
                unittest.mock.MagicMock(),
                [
                    DataCollectionInfo(
                        data_collection_type=True,
                        data_collection_uuid="to_ingest_video_id_1_0000",
                        is_test=True,
                    ),
                    DataCollectionInfo(
                        data_collection_type=True,
                        data_collection_uuid="to_ingest_video_id_2_0000",
                        is_test=True,
                    ),
                ],
            )
            scale_task_summary = ScaleTaskSummary(
                video_uuids=[
                    vid.data_collection_uuid for vid in videos_to_ingest
                ],
                fps=0,
                task_type="scale_task_type",
                batch_name="my_batch_name",
                data_type=DataCollectionType.VIDEO,
                task_unique_ids=[
                    "scale_uuid_1",
                    "scale_uuid_2",
                    "scale_uuid_3",
                    "scale_uuid_4",
                ],
                generate_hypothesis=False,
            )
            func_mocks[
                create_scale_tasks
            ].mock.return_value = scale_task_summary
            func_mocks[
                ingest_data_collections_to_metaverse
            ].mock.return_value = (
                [vid.data_collection_uuid for vid in videos_to_ingest][1:],
                [videos_to_ingest[0].data_collection_uuid],
            )
            # trunk-ignore(pylint/E1101)
            result: IngestionSummary = door_dataflywheel(
                camera_uuid="americold/ontario/0005/cha",
                start_date=None,
                max_incidents=1,
                metaverse_environment="INTERNAL",
                overwrite_config_file=None,
                pipeline_setup=PipelineSetup(),
            ).resolve(SilentResolver())
            self.assertIsInstance(result, IngestionSummary)
            assert result.metaverse_environment == "INTERNAL"
            chunk_input = func_mocks[
                chunk_and_upload_video_logs
            ].mock.call_args[1]["ingest_input"]
            assert (
                chunk_input.metadata
                == [
                    DataCollectionInfo(
                        data_collection_type=DataCollectionType.VIDEO,
                        data_collection_uuid="to_ingest_video_id_1",
                        is_test=True,
                    ),
                    DataCollectionInfo(
                        data_collection_type=DataCollectionType.VIDEO,
                        data_collection_uuid="to_ingest_video_id_2",
                        is_test=True,
                    ),
                ]
                != [
                    DataCollectionInfo(
                        data_collection_type=None,
                        data_collection_uuid="to_ingest_video_id_1",
                        is_test=True,
                    ),
                    DataCollectionInfo(
                        data_collection_type=None,
                        data_collection_uuid="to_ingest_video_id_2",
                        is_test=True,
                    ),
                ]
            )
