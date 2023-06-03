##
## Copyright 2020-2021 Voxel Labs, Inc.
## All rights reserved.
##
## This document may not be reproduced, republished, distributed, transmitted,
## displayed, broadcast or otherwise exploited in any manner without the express
## prior written permission of Voxel Labs, Inc. The receipt or possession of this
## document does not convey any rights to reproduce, disclose, or distribute its
## contents, or to manufacture, use, or sell anything that it may describe, in
## whole or in part.
##

import tempfile

import ffmpeg
from loguru import logger

from core.infra.cloud.gcs_utils import (
    download_video_object_from_cloud,
    upload_to_gcs,
)


def transcode_video_to_h264(video_uuid: str) -> None:
    """Transcode video uuid from voxel-logs to h264.

    Args:
        video_uuid (str): Video uuid of the video to transcode

    Raises:
        RuntimeError: Throws when the video entered is already h264.
    """
    with tempfile.TemporaryDirectory() as tempdir, tempfile.NamedTemporaryFile() as temp:
        output_path = download_video_object_from_cloud(
            video_uuid, "voxel-logs", tempdir
        )
        codec = ffmpeg.probe(output_path)["streams"][0]["codec_name"]
        if codec == "h264":
            raise RuntimeError("File already in h264 format")
        transcoded_filename = f"{temp.name}.mp4"
        transcode_to_h264(output_path, transcoded_filename)
        gcs_path = f"gs://voxel-logs-h264/{video_uuid}.mp4"
        upload_to_gcs(gcs_path, transcoded_filename, content_type="video/mp4")
        logger.info(f"Transcoded and uploaded to {gcs_path}")


def transcode_to_h264(input_file: str, output_file: str) -> None:
    """Transcodes input file to h264 output.

    Note: These videos should not be used for training since there is loss
    of frame information during transcoding.

    Args:
        input_file (str): File to transcode
        output_file (str): File transcoded to h264
    """
    command = (
        ffmpeg.input(input_file)
        .output(output_file, **{"vsync": "0", "c:v": "libx264"})
        .global_args("-nostdin")
        .global_args("-hide_banner")
    )
    raw_command = "ffmpeg " + " ".join(command.get_args())
    logger.debug(f"Running: '{raw_command}'")
    command.run(
        overwrite_output=True,
    )
    logger.warning(
        "H264 videos should not be used for training, since transcoding results in some loss of frame information"
    )
