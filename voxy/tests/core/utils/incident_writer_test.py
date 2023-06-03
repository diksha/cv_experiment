import os
import unittest
from unittest import mock

import boto3
from moto import mock_kinesisvideo

from core.infra.cloud import kinesis_utils
from core.structs.frame import Frame
from core.structs.incident import Incident
from core.structs.video import Video
from core.utils.incident_writer import (
    IncidentWriter,
    IncidentWriterIncident,
    IncidentWriterInput,
)


class IncidentWriterTest(unittest.TestCase):
    @mock_kinesisvideo
    def setUp(self):
        kvs_client = boto3.client("kinesisvideo", region_name="us-west-2")
        stream_name = "my-stream"
        stream = kvs_client.create_stream(StreamName=stream_name)
        self.incident_writer = IncidentWriter(
            input_params=IncidentWriterInput(
                temp_directory="/tmp",
                dry_run=False,
                camera_arn=stream["StreamARN"],
                video_uuid=None,
                fps=5,
            )
        )

    @mock_kinesisvideo
    def test_write_incident_thumbnail(self) -> None:
        with mock.patch.object(
            kinesis_utils, "extract_thumbnail", return_value=None
        ) as mock_method:
            self.incident_writer._write_incident_thumbnail(  # trunk-ignore(pylint/W0212)
                Incident(uuid="uuid")
            )
            mock_method.assert_called_once()

    @mock_kinesisvideo
    def test_write_incident_video(self) -> None:
        with mock.patch.object(
            kinesis_utils, "download_media", return_value=None
        ) as download_mock_method, mock.patch.object(
            kinesis_utils, "download_original_video", return_value=None
        ) as download_original_mock_method:
            self.incident_writer._write_incident_video(  # trunk-ignore(pylint/W0212)
                Incident(uuid="uuid"), Video(uuid="uuid"), []
            )
            download_mock_method.assert_called_once()
            download_original_mock_method.assert_called_once()

    def test_write_incident_annotations(self) -> None:
        video = Video(uuid="uuid")
        video.frames.append(
            Frame(
                frame_number=1,
                frame_width=1,
                frame_height=1,
                relative_timestamp_ms=1,
                epoch_timestamp_ms=1,
                relative_timestamp_s=1,
            )
        )
        self.incident_writer._write_incident_annotations(  # trunk-ignore(pylint/W0212)
            Incident(uuid="uuid"), video
        )
        assert os.path.exists("/tmp/uuid_annotations.json")

    def test_sort_incident_writer_incidents(self) -> None:
        incidents = [
            IncidentWriterIncident(
                incident=Incident(uuid="two"), end_time=2.0
            ),
            IncidentWriterIncident(
                incident=Incident(uuid="one"), end_time=1.0
            ),
        ]

        incidents.sort()
        assert incidents[0].incident.uuid == "one"
