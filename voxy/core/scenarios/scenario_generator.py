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
import atexit
import os
import random
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from core.infra.cloud import gcs_utils

ScenarioIdentifier = namedtuple(
    "ScenarioIdentifier",
    "start_timestamp scenario_duration incident_type scenario_type",
)


class ScenarioVideoGenerator:
    def __init__(self, video_format="mp4", buffer_s=0):
        self.temp_dir = tempfile.mkdtemp()
        self.video_format = video_format
        self.buffer_s = buffer_s
        atexit.register(self.cleanup)

    def _random_uuid(self, seed):
        random.seed(seed)
        return str(
            uuid.UUID(
                bytes=bytes(random.getrandbits(8) for _ in range(16)),
                version=4,
            )
        )

    def generate(
        self, video_uuid, scenario_identifiers, video_bucket="voxel-logs"
    ):
        generated_scenario_videos = []
        video_local_path = gcs_utils.download_video_object_from_cloud(
            video_uuid, video_bucket, output_dir=self.temp_dir
        )
        ffmpeg_template = "ffmpeg -ss {} -i {} -t {} -c copy {}"
        for _, scenario_identifier in enumerate(scenario_identifiers):
            # Lengthen the scenario if a buffer is specified.
            # Buffer is added at the start and end of a scenario.
            start_timestamp = datetime.strptime(
                scenario_identifier.start_timestamp, "%H:%M:%S"
            )
            start_timestamp = max(
                start_timestamp - timedelta(seconds=self.buffer_s),
                datetime.strptime("00:00:00", "%H:%M:%S"),
            )
            start_timestamp = start_timestamp.strftime("%H:%M:%S")
            duration = datetime.strptime(
                scenario_identifier.scenario_duration, "%H:%M:%S"
            ) + timedelta(seconds=self.buffer_s)
            duration = duration.strftime("%H:%M:%S")
            video_name = gcs_utils.video_name_from_uuid(video_uuid)
            output_dir = os.path.join(
                self.temp_dir,
                os.path.dirname(video_uuid),
                "scenarios",
                scenario_identifier.incident_type,
                scenario_identifier.scenario_type,
            )
            # seed to make the uuid deterministic
            seed = video_uuid + str(start_timestamp) + str(duration)
            video_name = self._random_uuid(seed)
            output_file_name = video_name + "." + self.video_format
            # ffmpeg can't create directories on it's own
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(output_dir, output_file_name)
            ffmpeg_command = ffmpeg_template.format(
                start_timestamp,
                video_local_path,
                duration,
                output_path,
            )

            ret = os.system(ffmpeg_command)
            if ret == 0:
                generated_scenario_videos.append(output_path)

        return generated_scenario_videos

    def upload_to_gcs(
        self, local_paths: List, output_bucket="voxel-logs"
    ) -> List:
        """Uploads the files to gcs

        Args:
            local_paths (List): paths to upload files from
            output_bucket (str, optional): output bucket to upload files to.
            Defaults to "voxel-logs".

        Returns:
            List: videos uploaded
        """
        uploaded_videos = []
        storage_client = gcs_utils.get_storage_client()
        for local_path in local_paths:
            cloud_path = os.path.normpath(
                output_bucket + "/" + local_path.replace(self.temp_dir, "")
            )
            gcs_path = "gs://" + cloud_path
            gcs_utils.upload_to_gcs(
                gcs_path,
                local_path,
                content_type="video/" + self.video_format,
                storage_client=storage_client,
            )
            uploaded_videos.append(gcs_path)
        return uploaded_videos

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class ScenariosSource(ABC):
    def __init__(self, resource_path):
        self.resource_path = resource_path

    @abstractmethod
    def extract_scenarios(self):
        pass
