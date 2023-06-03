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
import json
import os
from fractions import Fraction

import av
import cv2

from core.infra.cloud.gcs_utils import (
    get_video_signed_url,
    read_from_gcs,
    upload_to_gcs,
)


def main():
    filename = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"],
        "data",
        "scenario_sets",
        "americold",
        "modesto",
        "first.json",
    )
    scenarios = json.loads(open(filename).read())

    for scenario in scenarios["scenarios"]:
        video_uuid = scenario["video_uuid"]
        # camera_uuid = scenario["camera_uuid"]
        incident_uuid = video_uuid.split("/")[-1].replace(".mp4", "")
        annotation_path = (
            f"gs://voxel-portal/incidents/{incident_uuid}_annotations.json"
        )
        annotations = json.loads(read_from_gcs(annotation_path))

        # video_path = f"gs://voxel-portal/incidents/{incident_uuid}_video.mp4"
        updated_video_path = f"gs://voxel-logs/{video_uuid}.mp4"
        frames = []
        cap = cv2.VideoCapture(
            get_video_signed_url(
                bucket="voxel-portal",
                video_uuid=f"incidents/{incident_uuid}_video",
            )
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        print(video_uuid, len(frames), len(annotations["frames"]))
        # assert len(frames) == len(annotations['frames'])

        temp_filename = f"{incident_uuid}.mp4"
        temp_filepath = os.path.join("/tmp/", temp_filename)

        sorted_timestamps = sorted(
            [frame["relative_timestamp_ms"] for frame in annotations["frames"]]
        )
        start_ts_ms = sorted_timestamps[0]
        sorted_relative_timestamps = [
            item - start_ts_ms for item in sorted_timestamps
        ]

        container = av.open(temp_filepath, mode="w")
        stream = container.add_stream("h264", rate=10)
        stream.codec_context.time_base = Fraction(1, 1000)

        stream.width = 960
        stream.height = 480
        stream.pix_fmt = "yuv420p"

        for idx in range(0, len(frames)):
            relative_timestamp_ms = sorted_relative_timestamps[idx]
            img = frames[idx]

            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = int(relative_timestamp_ms)

            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush fstream
        for packet in stream.encode():
            container.mux(packet)

        # Close the file
        container.close()

        upload_to_gcs(
            updated_video_path, temp_filepath, content_type="video/mp4"
        )


main()
