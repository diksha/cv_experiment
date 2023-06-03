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

import mock
from sematic.resolvers.silent_resolver import SilentResolver

from core.labeling.logs_store.chunk_and_upload_video_logs import (
    VideoIngestInput,
    chunk_and_upload_video_logs,
    create_video_ingest_input,
)
from core.labeling.logs_store.ingest_data_collection_metaverse import (
    DataCollectionInfo,
)
from core.structs.data_collection import DataCollectionType


class ChunkAndUploadVideoLogsTest(unittest.TestCase):
    @mock.patch(
        "core.labeling.logs_store.chunk_and_upload_video_logs."
        "aws_utils.download_video_object_from_cloud"
    )
    @mock.patch(
        "core.labeling.logs_store.chunk_and_upload_video_logs.upload_file"
    )
    def test_chunk_and_upload_video(
        self,
        mock_upload_file: mock.Mock,
        mock_download_video_object: mock.Mock,
    ) -> None:
        """Tests input and output of chunk logs

        Args:
            mock_upload_file (mock.Mock): mocks aws upload
            mock_download_video_object (mock.Mock): mocks aws download
        """

        mock_download_video_object.return_value = (
            "tests/core/labeling/logs_store/20220606_01_doors.mp4"
        )
        mock_upload_file.return_value = (
            "s3://voxel-logs/americold/modesto/0001/cha/1.mp4"
        )
        video_uuids = [
            "americold/modesto/0001/cha/1",
        ]

        # trunk-ignore(pylint/E1101)
        video_ingest_input = create_video_ingest_input(
            video_uuids=video_uuids,
        ).resolve(SilentResolver())

        self.assertEqual(
            VideoIngestInput(
                video_uuids=[
                    "americold/modesto/0001/cha/1",
                ],
                input_source="s3",
                input_bucket="voxel-raw-logs",
                input_prefix=None,
                max_video_chunk_size_s=600,
                metadata=None,
            ),
            video_ingest_input,
        )

        self.assertEqual(
            # trunk-ignore(pylint/E1101)
            chunk_and_upload_video_logs(video_ingest_input).resolve(
                SilentResolver()
            ),
            (
                ["americold/modesto/0001/cha/1"],
                [
                    DataCollectionInfo(
                        data_collection_uuid="americold/modesto/0001/cha/1",
                        is_test=False,
                        data_collection_type=DataCollectionType.VIDEO,
                    )
                ],
            ),
        )
