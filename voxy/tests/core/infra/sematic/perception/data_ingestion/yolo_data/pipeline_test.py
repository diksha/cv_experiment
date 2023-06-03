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
import typing

# lightly hack to avoid running network call upon import
# https://github.com/lightly-ai/lightly/blob/fc252424afae73af54826d97b36042130522d025/lightly/__init__.py#L126
# trunk-ignore-all(pylint/C0413)
# trunk-ignore-all(flake8/E402)
os.environ["LIGHTLY_DID_VERSION_CHECK"] = "True"
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.perception.data_ingestion.yolo_data.pipeline import (
    IngestionSummary,
    LightlyIngestionUserInput,
    get_input_bucket,
    ingest_object_detection_data,
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
from core.labeling.tools.pull_kinesis_feed import PullFeedResult
from core.labeling.tools.pull_kinesis_feed_site import pull_kinesis_feed_site
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
from core.utils.yaml_jinja import load_yaml_with_jinja


class PipelineTest(unittest.TestCase):
    """Tests for the YOLO Data ingestion pipeline"""

    @patch(
        "core.ml.data.curation.check_location_site.organization_has_cameras_online"
    )
    def test_step_connections(self, mock_cameras_online: Mock):
        """
        Confirm that the pipeline produces no errors and a summary of the ingestion

        Args:
            mock_cameras_online (Mock): mock function whether cameras are online
        """
        mock_cameras_online.return_value = True
        pipeline_funcs = [
            pull_kinesis_feed_site,
            prepare_lightly_run,
            trim_lightly_clips,
            chunk_and_upload_video_logs,
            create_scale_tasks,
            ingest_data_collections_to_metaverse,
            run_lightly_worker,
        ]
        with mock_sematic_funcs(funcs=pipeline_funcs) as func_mocks:
            func_mocks[pull_kinesis_feed_site].mock.return_value = [
                PullFeedResult(
                    "americold/modesto/0001/cha",
                    "s3://kinesis-pull-bucket/kinesis/pull/path",
                )
            ]
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
            scale_task_summary = ScaleTaskSummary(
                video_uuids=[
                    vid.data_collection_uuid for vid in videos_to_ingest
                ],
                fps=0,
                task_type="scale_task_type",
                data_type=DataCollectionType.VIDEO,
                batch_name="my_batch_name",
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

            # trunk-ignore(pylint/E1101)
            result: IngestionSummary = ingest_object_detection_data(
                organization="americold",
                location="ontario",
                ingestion_datetime=datetime.now(),
                metaverse_environment="INTERNAL",
                config=load_yaml_with_jinja(
                    "core/infra/sematic/perception/data_ingestion/yolo_data/configs/production.yaml"
                ),
                pipeline_setup=PipelineSetup(),
                test_size=0.2,
                camera_batch_map=LightlyIngestionUserInput(
                    lightly_num_samples=1000,
                    specific_camera_uuids=[
                        "americold/ontario/0001/cha",
                        "americold/ontario/0002/cha",
                        "americold/ontario/0003/cha",
                    ],
                ),
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

            # trunk-ignore(pylint/E1101)
            result_default: IngestionSummary = ingest_object_detection_data(
                organization="americold",
                location="ontario",
                ingestion_datetime=datetime.now(),
                metaverse_environment="INTERNAL",
                config=load_yaml_with_jinja(
                    "core/infra/sematic/perception/data_ingestion/yolo_data/configs/production.yaml"
                ),
                pipeline_setup=PipelineSetup(),
                test_size=0.2,
                camera_batch_map=LightlyIngestionUserInput(
                    lightly_num_samples=-1,
                    specific_camera_uuids=None,
                ),
            ).resolve(SilentResolver())
            self.assertIsInstance(result_default, IngestionSummary)
            assert result_default.metaverse_environment == "INTERNAL"
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

    def test_feeds_available(self) -> None:
        """
        Tests if the pipeline raises when all feeds are not available
        """

        def test_pull_feed_result(
            uuid: str, s3_path: typing.Optional[str]
        ) -> PullFeedResult:
            """
            Dummy mock pull feed result

            Args:
                uuid: the uuid to find
                s3_path: the s3 path

            Returns:
                PullFeedResult: the dummy feed result
            """
            return PullFeedResult(uuid, s3_path)

        fake_cameras = [f"acme/nyc/{i}/cha" for i in range(5)]
        all_failed = []

        for cam in fake_cameras:
            all_failed.append(test_pull_feed_result(cam, None))

        # test if empty
        with self.assertRaisesRegex(
            Exception, ".*RuntimeError: All kinesis streams are down.*"
        ):
            get_input_bucket([]).resolve(SilentResolver())

        with self.assertRaisesRegex(
            Exception, ".*RuntimeError: All kinesis streams are down.*"
        ):
            get_input_bucket(all_failed).resolve(SilentResolver())
        some_failed = all_failed
        uuid = "foo/bar/001/cha"
        some_failed.append(
            test_pull_feed_result(
                uuid, f"s3://voxel-lightly-input/door/{uuid}/hi.mp4"
            )
        )

        # there is no assert no raise
        self.assertTrue(
            get_input_bucket(some_failed).resolve(SilentResolver()) is not None
        )
