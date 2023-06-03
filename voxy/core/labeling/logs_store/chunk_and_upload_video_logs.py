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

"""
Chunks the video in 10 minute long videos and stores it in google cloud.
"""
import datetime
import glob
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import ffmpeg
import sematic
from loguru import logger

from core.labeling.constants import (
    H265_CODEC,
    VOXEL_RAW_BUCKET,
    VOXEL_VIDEO_LOGS_BUCKET,
)
from core.labeling.logs_store.ingest_data_collection_metaverse import (
    DataCollectionInfo,
)
from core.labeling.logs_store.ingestion_helpers import (
    validate_voxel_uuid_format,
)
from core.structs.data_collection import DataCollectionType
from core.utils import aws_utils
from core.utils.aws_utils import get_bucket_path_from_s3_uri, upload_file

# Example commands:
# ./bazel run core/labeling/logs_store:chunk_and_upload_video_logs --
# --src s3 --bucket voxel-lightly-input
# --videos americold/ontario/0005/cha/2022-06-26_1656201600000_ms_1656202200000_ms

VIDEO_FORMAT = "mp4"


@dataclass
class VideoIngestInput:
    """Input to Video Ingest Pipeline"""

    video_uuids: List[str]
    input_source: str = "s3"
    input_bucket: str = VOXEL_RAW_BUCKET
    input_prefix: Optional[str] = None
    max_video_chunk_size_s: int = 600
    metadata: Optional[List[DataCollectionInfo]] = None


# trunk-ignore-begin(pylint/W9011,pylint/W9015)
@sematic.func
def create_video_ingest_input(
    video_uuids: List[str],
    input_source: str = "s3",
    input_bucket: str = VOXEL_RAW_BUCKET,
    input_prefix: Optional[str] = None,
    max_video_chunk_size_s: int = 600,
    metadata: Optional[List[DataCollectionInfo]] = None,
) -> VideoIngestInput:
    """# Creates a VideoIngestInput

    Made as a Sematic func so the object can be created using futures

    ## Parameters
    - **video_uuids**:
        A list of videos to ingest
    - **input_source**:
        "s3", depending on where data is coming from
    - **input_bucket**:
        The bucket videos are coming from
    - **input_prefix**:
        An optional prefix between the bucket and video ids
    - **max_video_chunk_size_s**:
        Max size of the video chunks in seconds
    - **metadata**:
        Metadata to associate with the videos

    ## Returns
    An instance of VideoIngestInput
    """
    # trunk-ignore-end(pylint/W9011,pylint/W9015)
    return VideoIngestInput(
        video_uuids=video_uuids,
        input_source=input_source,
        input_bucket=input_bucket,
        input_prefix=input_prefix,
        max_video_chunk_size_s=max_video_chunk_size_s,
        metadata=metadata,
    )


def is_video_valid(video_path: str) -> bool:
    """Validates that videos to be ingested follow certain standards.
    Current checks:
        1. Video is of H265 format

    Args:
        video_path : Path to local video file

    Returns:
        Whether the video passes the validation checks.
    """
    codec = ffmpeg.probe(video_path)["streams"][0]["codec_name"]
    if codec != H265_CODEC:
        logger.error(f"Format of video should be h265 found {codec}")
        return False
    return True


def create_video_chunks(
    video_uuid: str,
    output_dir: str,
    input_src: str,
    input_cloud_bucket: str,
    input_prefix: str,
    chunk_size_s: int = 600,
) -> str:
    """
    Given a video uuid, chunk the video such that each chunk length is
    smaller or equal to the chunk_size_s.

    Args:
        video_uuid : an identifier that uniquely identifies a video in S3
        output_dir : local directory to store any downloaded videos
        input_src : Whether to download the video from S3
        input_cloud_bucket : The bucket that contains the video
        input_prefix : The path prefix where the video is stored
        chunk_size_s : The maximum length of any video chunks generated

    Returns:
        Path to a local directory where the video chunks are stored

    Raises:
        RuntimeError: If input source is incorrectly specified or if the video cannot be found
    """
    if input_src == "s3":
        video_path = aws_utils.download_video_object_from_cloud(
            video_uuid,
            input_cloud_bucket,
            output_dir,
            input_prefix=input_prefix,
        )
    else:
        raise RuntimeError("Input source can only be s3")
    if not is_video_valid(video_path):
        raise RuntimeError(f"Video is not valid for {video_uuid}")

    segment = str(datetime.timedelta(seconds=chunk_size_s))
    output_path, old_name = os.path.split(video_path)
    new_name = old_name.split(".")[0] + "_%04d" "." + VIDEO_FORMAT
    output_path = os.path.join(output_path, new_name)
    ffmpeg.input(video_path).output(
        output_path,
        v=1,
        c="copy",
        map=0,
        reset_timestamps=1,
        f="segment",
        segment_time=segment,
        segment_format="mp4",
    ).run()
    os.remove(video_path)
    logger.info(f"outputpath is {output_path}")
    return output_path


# trunk-ignore-begin(pylint/W9011,pylint/W9015)
@sematic.func
def chunk_and_upload_video_logs(
    ingest_input: VideoIngestInput,
    output_store_uuid: Optional[str] = None,
) -> Tuple[List[str], List[DataCollectionInfo]]:
    """# Chunks the videos and uploads to s3.

    ## Parameters
    - **ingest_input**:
        A VideoIngestInput object containing videos to ingest and other relevant metadata.
    - **output_store_uuid**:
        An id to use to ingest to

    ## Returns
    Tuple of: 1. uploaded video files, 2. ingestion metadata
    """

    def clean_temp_dir(output_dir, video_uuid):
        video_dir = os.path.split(f"{output_dir}/{video_uuid}")[0]
        filenames = glob.glob(os.path.join(video_dir, "*"), recursive=True)
        for filename in filenames:
            os.remove(filename)

    # trunk-ignore-end(pylint/W9011,pylint/W9015)
    validate_voxel_uuid_format(ingest_input.video_uuids)

    output_cloud_bucket = VOXEL_VIDEO_LOGS_BUCKET
    uploaded_videos = []
    output_metadata = []

    metadata_map = {
        video.data_collection_uuid: video
        for video in (ingest_input.metadata or [])
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        for video_uuid in ingest_input.video_uuids:
            clean_temp_dir(temp_dir, video_uuid)
            output_video_path = create_video_chunks(
                video_uuid,
                input_src=ingest_input.input_source,
                input_cloud_bucket=ingest_input.input_bucket,
                input_prefix=ingest_input.input_prefix,
                output_dir=temp_dir,
                chunk_size_s=ingest_input.max_video_chunk_size_s,
            )
            if output_video_path == "":
                continue
            video_dir = os.path.split(output_video_path)[0]
            s3_directory = os.path.relpath(video_dir, temp_dir)
            filenames = glob.glob(os.path.join(video_dir, "*"), recursive=True)
            mp4_filenames = [
                filename for filename in filenames if filename.endswith(".mp4")
            ]
            for local_file in mp4_filenames:
                filename = os.path.relpath(local_file, video_dir)
                uploaded_file = upload_file(
                    output_cloud_bucket,
                    local_file,
                    os.path.join(s3_directory, filename),
                )
                uploaded_video = os.path.splitext(
                    get_bucket_path_from_s3_uri(uploaded_file)[1]
                )[0]
                output_metadata_for_uuid = DataCollectionInfo(
                    data_collection_uuid=uploaded_video,
                    is_test=metadata_map.get(video_uuid).is_test
                    if metadata_map.get(video_uuid)
                    else False,
                    data_collection_type=metadata_map[
                        video_uuid
                    ].data_collection_type
                    if metadata_map.get(video_uuid)
                    else DataCollectionType.VIDEO,
                )
                output_metadata.append(output_metadata_for_uuid)
                uploaded_videos.append(uploaded_video)

    return (
        uploaded_videos,
        output_metadata,
    )
