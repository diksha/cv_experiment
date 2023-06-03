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
import io
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import av
import boto3
import botocore
import botocore.exceptions
import cv2
from botocore.stub import Stubber
from loguru import logger
from rules_python.python.runfiles import runfiles

from core.infra.cloud import kinesis_utils
from core.infra.cloud.kinesis_utils import KinesisVideoMediaReader
from core.utils.pyav_decoder import PyavDecoder

MKV_TEST_FILE = "kinesis_utils_testdata/testfile.mkv"
MP4_TEST_FILE = "kinesis_utils_testdata/testfile.mp4"

################# Helpers ##############


def _load_testdata(filename):
    with open(
        runfiles.Create().Rlocation(filename),
        "rb",
    ) as f:
        return f.read(-1)


def _mkv_testfile_data():
    return _load_testdata(MKV_TEST_FILE)


def _mp4_testfile_data():
    return _load_testdata(MP4_TEST_FILE)


def _get_timestamps(video_data):
    timestamps = []
    with PyavDecoder(io.BytesIO(video_data)) as decoder:
        for frame in decoder:
            timestamps.append(frame.timestamp_ms())
        timestamps.sort()
        return timestamps


def _get_media_data_duration_ms(video_data: bytes):
    start_ts, end_ts = None, None
    with av.open(io.BytesIO(video_data)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            pts = stream.time_base * 1000 * packet.pts
            if start_ts is None or start_ts > pts:
                start_ts = pts
            if end_ts is None or pts > end_ts:
                end_ts = pts
    return end_ts - start_ts


def _get_media_file_duration_ms(filename: str):
    with open(filename, "rb") as f:
        return _get_media_data_duration_ms(f.read(-1))


MKV_TEST_FILE_DATA = _mkv_testfile_data()
MKV_TEST_FILE_TIMESTAMPS = _get_timestamps(MKV_TEST_FILE_DATA)
MKV_TEST_FILE_FRAME_COUNT = len(MKV_TEST_FILE_TIMESTAMPS)


class MockKVS:
    def __init__(self, mock_client, video_data=None):
        """Generates a mocked up boto3 client for kinesisservice and kinesis-video-media

        Args:
            mock_client: a unittest.mock.patch object for boto3.client
        """
        self._kv_client = botocore.session.get_session().create_client(
            "kinesisvideo",
            region_name="us-west-2",
        )
        self._kv_stubber = Stubber(self._kv_client)

        self._kvm_client = botocore.session.get_session().create_client(
            "kinesis-video-media",
            region_name="us-west-2",
        )
        self._kvm_stubber = Stubber(self._kvm_client)

        mock_client.side_effect = self._boto3_client

    def _boto3_client(self, service, *args, **kwargs):
        if service == "kinesisvideo":
            return self._kv_client
        if service == "kinesis-video-media":
            return self._kvm_client
        raise RuntimeError(
            "unsupported service name passed to mock boto3 client"
        )

    def activate(self):
        self._kv_stubber.activate()
        self._kvm_stubber.activate()

    def assert_no_pending_responses(self):
        self._kv_stubber.assert_no_pending_responses()
        self._kvm_stubber.assert_no_pending_responses()

    def add_get_data_endpoint_response(
        self, service_response, expected_params=None
    ):
        self._kv_stubber.add_response(
            "get_data_endpoint", service_response, expected_params
        )

    def add_default_get_data_endpoint_response(self):
        self.add_get_data_endpoint_response(
            {
                "DataEndpoint": "some-fake-endpoint",
            }
        )

    def add_get_data_endpoint_error(self):
        self._kv_stubber.add_client_error("get_data_endpoint")

    def add_get_media_response(
        self, payload: bytes = MKV_TEST_FILE_DATA, expected_params=None
    ):
        self._kvm_stubber.add_response(
            "get_media",
            {
                "ContentType": "video/matroska",
                "Payload": botocore.response.StreamingBody(
                    io.BytesIO(payload),
                    len(
                        payload,
                    ),
                ),
            },
            expected_params,
        )

    def add_get_media_error(self):
        self._kvm_stubber.add_client_error("get_media")


############ End Helpers ###########


# TODO: make these tests a little faster by pre-caching more
#       of the basic metadata needed for the tests
class KinesisUtilsTest(unittest.TestCase):
    def test_testdata(self):
        self.assertGreater(len(MKV_TEST_FILE_DATA), 0)

        with av.open(io.BytesIO(MKV_TEST_FILE_DATA)) as container:
            stream = container.streams.video[0]
            for _ in container.demux(stream):
                continue

    @patch.object(boto3, "client")
    def test_kinesis_stream_reader(self, mock_client) -> None:
        kvs = MockKVS(mock_client)
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response()
        kvs.activate()

        frame_count = 0
        with KinesisVideoMediaReader(stream_arn="some-fake-arn") as ksr:
            for _ in ksr:
                frame_count += 1

        self.assertEqual(
            frame_count,
            MKV_TEST_FILE_FRAME_COUNT,
            "frame count should match source content",
        )

        kvs.assert_no_pending_responses()

    @patch.object(boto3, "client")
    def test_kinesis_stream_reader_error(self, mock_client):
        kvs = MockKVS(mock_client)
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_error()
        kvs.activate()

        self.assertRaises(
            kinesis_utils.KinesisVideoError,
            kinesis_utils.KinesisVideoMediaReader,
            "some-fake-arn",
        )

    @patch.object(boto3, "client")
    def test_kinesis_stream_run_once_false(self, mock_client):
        kvs = MockKVS(mock_client)
        # expect three requests, the first two should be successful and the last should error out
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response()
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response()
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_error()
        kvs.activate()

        frame_count = 0
        with KinesisVideoMediaReader(
            stream_arn="some-fake-arn", run_once=False
        ) as stream:
            try:
                for _ in stream:
                    frame_count += 1
            except kinesis_utils.KinesisVideoError:
                pass

        self.assertEqual(
            frame_count,
            MKV_TEST_FILE_FRAME_COUNT * 2,
            "should have two testfiles worth of frames",
        )

        # this is the important assert here, ensures that get_media was called twice
        kvs.assert_no_pending_responses()

    @patch.object(boto3, "client")
    def test_download_media_start_time(self, mock_client) -> None:
        sorted_timestamps = MKV_TEST_FILE_TIMESTAMPS[10:-10]
        kvs = MockKVS(mock_client)
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response(
            expected_params={
                "StreamARN": "some-fake-arn",
                "StartSelector": {
                    "StartSelectorType": "PRODUCER_TIMESTAMP",
                    "StartTimestamp": datetime.fromtimestamp(
                        sorted_timestamps[0] / 1000.0
                    )
                    - timedelta(
                        seconds=kinesis_utils._KINESIS_CHUNK_DURATION_S  # trunk-ignore(pylint/W0212)
                    ),
                },
            }
        )
        kvs.activate()

        with tempfile.TemporaryDirectory() as td:
            filepath = os.path.join(td, "test-media.mp4")
            kinesis_utils.download_media(
                camera_arn="some-fake-arn",
                sorted_timestamps=sorted_timestamps,
                filepath=filepath,
                fps=5,
            )

            target_duration = sorted_timestamps[-1] - sorted_timestamps[0]
            duration = _get_media_file_duration_ms(filepath)
            self.assertEqual(
                duration,
                target_duration,
                "output duration should match input duration",
            )

        kvs.assert_no_pending_responses()

    @patch.object(boto3, "client")
    def test_kinesis_stream_reader_no_start_time(self, mock_client) -> None:
        kvs = MockKVS(mock_client)
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response(
            expected_params={
                "StreamARN": "some-fake-arn",
                "StartSelector": {
                    "StartSelectorType": "NOW",
                },
            }
        )
        kvs.activate()

        with KinesisVideoMediaReader(stream_arn="some-fake-arn") as stream:
            for frame in stream:
                self.assertIsNotNone(frame, "frame should not be none")
                return

        kvs.assert_no_pending_responses()

    @patch.object(boto3, "client")
    def test_kinesis_stream_reader_fps(self, mock_client) -> None:
        kvs = MockKVS(mock_client)
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response()
        kvs.activate()

        count = 0
        start_ts = None
        with KinesisVideoMediaReader(
            stream_arn="some-fake-arn", fps=5
        ) as stream:
            for frame in stream:
                count += 1

                if start_ts is None:
                    start_ts = frame.timestamp_ms()
                elif frame.timestamp_ms() - start_ts >= 5000:
                    break

        # we expect to have 25 frames but as timestamps are not exact we are unlikely to have precisely 25 frames
        self.assertGreater(
            count,
            20,
            "should have greater than 20 frames for 5s of 5fps video",
        )
        self.assertLess(
            count, 30, "should have less than 30 frames for 5s of 5fps video"
        )

        kvs.assert_no_pending_responses()

    @patch.object(boto3, "client")
    def test_download_media_and_thumbnail(self, mock_client) -> None:
        kvs = MockKVS(mock_client)
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response(payload=MKV_TEST_FILE_DATA)
        kvs.activate()

        # grab a chunk of timestamps out of the middle
        sorted_timestamps = MKV_TEST_FILE_TIMESTAMPS[10:-10]
        print("SORTED_TIMESTAMPS=", sorted_timestamps)

        with tempfile.TemporaryDirectory() as td:
            filepath = os.path.join(td, "test-media.mp4")

            kinesis_utils.download_media(
                camera_arn="some-fake-arn",
                sorted_timestamps=sorted_timestamps,
                filepath=filepath,
                fps=5,
            )

            target_duration = sorted_timestamps[-1] - sorted_timestamps[0]
            duration = _get_media_file_duration_ms(filepath)
            # check to make sure our duration is close to target
            self.assertEqual(
                duration,
                target_duration,
                "output duration should equal input duration",
            )

            thumbnail_path = os.path.join(td, "test-thumb.jpg")
            kinesis_utils.extract_thumbnail(
                inputpath=filepath, outputpath=thumbnail_path
            )
            thumbnail_shape = cv2.imread(thumbnail_path).shape

            self.assertEqual(
                thumbnail_shape[0], 480, "thumbnail height should be correct"
            )
            self.assertEqual(
                thumbnail_shape[1], 853, "thumbnail width should be correct"
            )

        kvs.assert_no_pending_responses()

    @patch.object(boto3, "client")
    def test_kinesis_stream_reader_empty_media_payload(
        self, mock_client
    ) -> None:
        # this test make sure that the reader does not fail on empty payload responses
        # which can happen when there is a stall in the producer
        kvs = MockKVS(mock_client)
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response(payload=MKV_TEST_FILE_DATA)
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response(payload=bytes())
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response(payload=MKV_TEST_FILE_DATA)
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_error()
        kvs.activate()

        frame_count = 0
        try:
            with KinesisVideoMediaReader(
                stream_arn="some-fake-arn",
                run_once=False,
            ) as stream:
                for _ in stream:
                    frame_count += 1
        except kinesis_utils.KinesisVideoError as e:
            logger.debug(f"stream ended with {e}")

        self.assertEqual(
            frame_count,
            MKV_TEST_FILE_FRAME_COUNT * 2,
            "should have two testfiles worth of frames",
        )

        kvs.assert_no_pending_responses()

    @patch.object(boto3, "client")
    def test_kinesis_stream_reader_stream_name(self, mock_client) -> None:
        kvs = MockKVS(mock_client)
        kvs.add_default_get_data_endpoint_response()
        kvs.add_get_media_response(
            expected_params={
                "StartSelector": {"StartSelectorType": "NOW"},
                "StreamName": "some-fake-stream-name",
            }
        )
        kvs.activate()

        with kinesis_utils.KinesisVideoMediaReader(
            stream_name="some-fake-stream-name",
        ) as stream:
            for _ in stream:
                pass

        kvs.assert_no_pending_responses()


if __name__ == "__main__":
    unittest.main()
