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

import argparse
import datetime
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import List, Optional

import ffmpeg
import sematic
from lightly.api import ApiWorkflowClient
from loguru import logger
from sematic.resolvers.silent_resolver import SilentResolver
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from core.infra.sematic.shared.resources import CPU_8CORE_32GB
from core.labeling.logs_store.ingest_data_collection_metaverse import (
    DataCollectionInfo,
)
from core.ml.data.curation.lib.lightly_worker import output_frame_sequence
from core.ml.data.curation.voxel_lightly_utils import (
    LIGHTLY_TOKEN_ARN,
    LightlyVideoFrameSequence,
)
from core.structs.data_collection import DataCollectionType
from core.utils import aws_utils


@dataclass
class TrimmedVideoSummary:
    """A summary of the results of trimming videos

    Attributes
    ----------
    output_bucket:
        The bucket where the resulting videos were written
    video_uuids:
        UUIDs of the trimmed videos
    failed_video_names:
        The names of videos that failed to trim/upload
    to_ingest_videos:
        The videos to ingest
    """

    output_bucket: str
    video_uuids: List[str]
    failed_video_names: List[str]
    to_ingest_videos: List[DataCollectionInfo]


def get_videos_to_ingest(
    video_uuids: list,
    video_uuid_to_test: dict,
    task: str,
) -> List[DataCollectionInfo]:
    """Get list of videos to ingest

    Args:
        video_uuids: a list of video uuids
        video_uuid_to_test: a dict mapping video name to whether it is to be used for test
        task: the task associated with the video intended for ingestion

    Returns:
        A list of the videos that are intended for ingestion
    """
    videos_to_ingest = []
    videos = []
    for video_uuid in video_uuids:
        is_test = video_uuid_to_test[video_uuid]
        videos_to_ingest.append(
            {
                "data_collection_uuid": video_uuid,
                "is_test": is_test,
                "data_collection_type": DataCollectionType.VIDEO.name,
            }
        )
        videos.append(
            DataCollectionInfo(
                data_collection_uuid=video_uuid,
                is_test=is_test,
                data_collection_type=DataCollectionType.VIDEO,
            )
        )
    n_true = sum(video.is_test for video in videos)
    n_videos = len(videos)
    logger.info(f"Number Test: {n_true}")
    logger.info(f"Number Train: {n_videos - n_true}")
    logger.info(f"Total: {n_videos}")
    logger.info(f"Proportion Train: { 1.0 - n_true / n_videos}")
    logger.info(f"Proportion Test: {n_true / n_videos}")
    return videos


def download_video(
    camera_uuid,
    video_name,
    input_bucket,
    local_filename,
    use_epoch_time,
    input_prefix=None,
):
    # Download file from S3
    filepath = os.path.join(camera_uuid, video_name)

    base = filepath.split(".")[0]
    basename = os.path.basename(base)
    print(filepath)
    ext = filepath.split(".")[1]
    if basename.count("_") > 4 or (
        not use_epoch_time and basename.count("_") > 1
    ):
        door_name_removed = "_".join(base.split("_")[:-2])
        filepath = ".".join([door_name_removed, ext])
    else:
        filepath = ".".join([base, ext])
    download_filepath = filepath
    if input_prefix is not None:
        download_filepath = os.path.join(input_prefix, filepath)

    logger.info(f"Downloading s3://{input_bucket}/{download_filepath}")

    aws_utils.download_to_file(input_bucket, download_filepath, local_filename)
    return filepath


def upload_video(output_bucket, local_video, video_uuid):
    s3_filename = video_uuid + ".mp4"
    aws_utils.upload_file(output_bucket, local_video, s3_filename)


# todo comments not allowed:
# trunk-ignore-all(pylint/W0511)
# TODO(twroge): this `get_video_uuid` logic will need to be updated when moved to sematic
#                where more context can be given and less assumptions are made. This should
#                all be removed
def get_video_uuid(  # trunk-ignore(pylint/R0913,pylint/R0914)
    local_video: str,
    video_name: str,
    start_pts: int,
    camera_uuid: str,
    task: str,
    use_epoch_timestamp: bool = False,
) -> str:
    """
    Generates a video uuid from the local video file, the original video name
    the start time, as well as the camera uuid. The output uuid will be something
    along the lines of

    organization/location/site/channel/task/custom_uuid

    the custom uuid will have information about the start time and duration

    Args:
        local_video (str): the local video file path
        video_name (str): the original video name that was downloaded
        start_pts (int): the start timestamp (offset time) in the video
        camera_uuid (str): the camera uuid for this video
        task (str): the task name to be ingested
        use_epoch_timestamp (bool, optional): Whether to interpret the video name as having
                                              the orignal epoch timestamps. Defaults to False.

    Raises:
        Exception: if the epoch timestamp format is incorrect

    Returns:
        str: the video uuid for this particular video
    """
    logger.info(f"Video name: {video_name}")
    duration = float(ffmpeg.probe(filename=local_video)["format"]["duration"])
    if round(duration) < 1:
        logger.error(f"Found zero length segment for video: {video_name} ")
        logger.error("Skipping.. ")
        return None

    if use_epoch_timestamp:
        if len(video_name.split("_")) < 4:
            raise Exception(
                "The video name is improperly formatted. Expected epoch timestamps"
            )
        epoch_start_ms = video_name.split("_")[1]
        epoch_video_start_s = int(epoch_start_ms) // 1000 + int(start_pts)
        epoch_video_end_s = epoch_video_start_s + round(duration)

        prepending = (
            "/".join(video_name.split("/")[:-1])
            if video_name.count("/") > 1
            else None
        )

        # Upload file
        video_uuid = (
            os.path.join(
                camera_uuid,
                prepending,
                task,
                f"{epoch_video_start_s}_{epoch_video_end_s}",
            )
            if prepending
            else os.path.join(
                camera_uuid,
                task,
                f"{epoch_video_start_s}_{epoch_video_end_s}",
            )
        )
    else:
        logger.info(video_name)
        # TODO(twroge): this needs to be updated when we move to sematic to support portal pipelines
        #              that are not door based
        base = os.path.splitext(os.path.basename(video_name))[0]
        door_postfix_present = base.count("_") > 0
        if door_postfix_present:
            door_name_removed = "_".join(base.split("_")[:-2])
            incident_uuid = door_name_removed
        else:
            incident_uuid = base

        logger.info(incident_uuid)
        relative_start_s = int(start_pts)
        relative_end_s = relative_start_s + round(duration)

        video_uuid = os.path.join(
            camera_uuid,
            task,
            f"{incident_uuid}_{relative_start_s}_{relative_end_s}",
        )
    logger.info(f"Video uuid: {video_uuid}")
    return video_uuid


def trim_sequence(
    sequence: dict,
    camera_uuid: str,
    input_bucket: str,
    output_bucket: str,
    task: str,
    remove_task_name: bool,
    use_epoch_time: bool,
    input_prefix: Optional[str] = None,
):
    """
    This trims an input sequence from lightly.

    NOTE: this is only intended to be used for short duration clips mainly for input
    into machine learning models where encoding is detrimental to video data. This should
    NOT be used for anything longer than a few minutes as the bit rate is around 4K video levels

    Args:
        sequence (dict): the lightly sequence dictionary
        camera_uuid (str): the camera uuid
        input_bucket (str): the input bucket to pull from
        output_bucket (str): the output bucket to write to
        task (str): the task to generate the output filenames
        remove_task_name (bool): whether to remove the task name in the uuid
        use_epoch_time (bool): whether to use the epoch timestamp
                               from the original video name (as with kinesis clips)
        input_prefix: Prefix between input_bucket and the camera_uuid/video_name to use
        when downloading videos. Defaults to None, in which case no prefix is used

    Raises:
        RuntimeError: when the intended trim sequence is too large

    Returns:
        tuple: video uuid and name
    """

    logger.warning(
        (
            "NOTE: this is only intended to be used for short duration clips "
            "mainly for inpu into machine learning models where encoding isnt "
            "detrimental to video data. This should NOT be used for anything "
            "longer than a few minutes as the bit rate is around 4K video levels"
        )
    )

    MAX_CLIP_DURATION_S = 30
    video_name = sequence["video_name"]
    pts_timestamps = sequence["frame_timestamps_pts"]
    logger.info(f"Processing {video_name}")
    # crop from start to stop pts
    with tempfile.NamedTemporaryFile(
        suffix=".mp4"
    ) as temp_video_file, tempfile.NamedTemporaryFile(
        suffix=".mp4"
    ) as temp_trim_file:
        download_video(
            camera_uuid,
            video_name,
            input_bucket,
            temp_video_file.name,
            use_epoch_time,
            input_prefix=input_prefix,
        )
        input_file_probe = ffmpeg.probe(filename=temp_video_file.name)
        # get timebase
        timebase = int(
            input_file_probe["streams"][0]["time_base"].split("/")[-1]
        )
        start_pts = min(pts_timestamps) / timebase
        end_pts = max(pts_timestamps) / timebase
        logger.info(f"Trimming: {video_name} from {start_pts} to {end_pts}")
        duration = end_pts - start_pts
        if duration > MAX_CLIP_DURATION_S:
            logger.error(f"Found long clip for: {video_name}")
            raise RuntimeError(
                "The duration of the clip was too large, this is intended for short duration clips only"
            )

        # Trim file
        command = ffmpeg.input(
            temp_video_file.name,
            ss=str(datetime.timedelta(seconds=start_pts)),
        ).output(
            temp_trim_file.name,
            t=str(datetime.timedelta(seconds=duration)),
            **{
                "c:v": "libx265",
                "x265-params": "lossless=1",
                "tag:v": "hvc1",
                "vsync": "vfr",
            },
        )
        raw_command = "ffmpeg " + " ".join(command.get_args())
        logger.debug(f"Running: '{raw_command}'")
        command.run(
            overwrite_output=True,
        )

        # get video names
        video_uuid = get_video_uuid(
            temp_trim_file.name,
            video_name,
            start_pts,
            camera_uuid,
            task,
            use_epoch_time,
        )
        if video_uuid is None:
            return None

        if remove_task_name:
            video_uuid = "/".join(video_uuid.split("/")[1:])

        upload_video(
            output_bucket,
            temp_trim_file.name,
            video_uuid,
        )
        return video_uuid


def perform_test_train_split(
    uuids: list, test_size: float = 0.2, seed: int = 42
) -> dict:
    """
    Generates a test train split of the input lightly sequences

    Args:
        uuids (list): the list of uuids to split
        test_size (float): proportion of test. Defaults to 0.2
        seed (int): random state of the test train split. Defaults to 42

    Returns:
        dict: a dictionary with the key being the uuid and a bool being if the value is test
    """
    train, test = train_test_split(
        uuids, test_size=test_size, random_state=seed
    )

    is_test_map = {}
    for item in train:
        is_test_map[item] = False

    for item in test:
        is_test_map[item] = True

    return is_test_map


# trunk-ignore-begin(pylint/W9015,pylint/W9011,pylint/R0913)
@sematic.func(
    resource_requirements=CPU_8CORE_32GB,
    standalone=True,
)
def trim_lightly_clips(
    input_bucket: str,
    output_bucket: str,
    camera_uuid: str,
    sequence_information: List[LightlyVideoFrameSequence],
    task: str,
    remove_task_name: bool,
    use_epoch_time: bool,
    test_size: float = 0.2,
    input_prefix: Optional[str] = None,
) -> TrimmedVideoSummary:
    """# Runs all steps of trimming and uploading clips.

    ## Parameters
    - **input_bucket**:
        Bucket of Lightly input videos to be trimmed
    - **output_bucket**:
        Bucket of Lightly output images after trimming
    - **camera_uuid**:
        Standard Voxel camera uuid
    - **sequence_information**:
        Information about the sequences identified by Lightly
    - **task**:
        The task name used as an output prefix.
    - **remove_task_name**:
        Whether or not to remove the task name from the resulting video UUID
    - **use_epoch_time**:
        Whether to use the epoch timestamp from the original video name
        (as with Kinesis clips)
    - **test_size**:
        Size of test train split
    - **input_prefix**:
        Prefix between input_bucket and the camera_uuid/video_name to use when
        downloading videos. Defaults to None, in which case no prefix is used

    ## Returns
    A TrimmedVideoSummary summarizing the result of the trimming

    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011,pylint/R0913)
    sequence_list = [asdict(sequence) for sequence in sequence_information]
    video_uuids = []
    failed_sequences = []
    if not sequence_list:
        return TrimmedVideoSummary(
            output_bucket=output_bucket,
            video_uuids=[],
            failed_video_names=[],
            to_ingest_videos=[],
        )

    for sequence in tqdm(sequence_list):
        try:
            trimmed_sequence = trim_sequence(
                sequence,
                camera_uuid,
                input_bucket,
                output_bucket,
                task,
                remove_task_name,
                use_epoch_time,
                input_prefix,
            )
        except RuntimeError as e:
            logger.error(e)
            failed_sequences.append(sequence["video_name"])
            continue

        if trimmed_sequence is None:
            failed_sequences.append(sequence["video_name"])
            continue

        video_uuids.append(trimmed_sequence)

    video_uuids = list(set(video_uuids))

    is_test_map = perform_test_train_split(video_uuids, test_size=test_size)

    to_ingest_videos = get_videos_to_ingest(
        video_uuids,
        is_test_map,
        task,
    )
    if failed_sequences:
        logger.error("Failed Sequences:")
        for sequence in failed_sequences:
            logger.error(f"Video: {sequence} failed")
    return TrimmedVideoSummary(
        output_bucket=output_bucket,
        video_uuids=video_uuids,
        failed_video_names=failed_sequences,
        to_ingest_videos=to_ingest_videos,
    )


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def get_output_bucket(trimmed_video_summary: TrimmedVideoSummary) -> str:
    """# Get the bucket trimmed videos were written to

    ## Parameters
    - **trimmed_video_summary**:
        Summary of a trimmed video task

    ## Returns
    The bucket trimmed videos were output to
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    return trimmed_video_summary.output_bucket


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def get_trimmed_video_uuids(
    trimmed_video_summary: TrimmedVideoSummary,
) -> List[str]:
    """# Get the uuids of trimmed videos

    ## Parameters
    - **trimmed_video_summary**:
        Summary of a trimmed video task

    ## Returns
    The UUIDs of trimmed videos
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    return trimmed_video_summary.video_uuids


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def get_to_ingest_videos(
    trimmed_video_summary: TrimmedVideoSummary,
) -> List[DataCollectionInfo]:
    """# Get the videos that should be ingested as a result of trimming

    ## Parameters
    - **trimmed_video_summary**:
        Summary of a trimmed video task

    ## Returns
    A list of videos to ingest
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    return trimmed_video_summary.to_ingest_videos


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--camera_uuid",
        required=True,
        type=str,
        help="The camera uuid to trim videos for",
    )
    parser.add_argument(
        "--lightly_run_id",
        required=True,
        type=str,
        help="The scheduled run id from lightly (can find in lightly logs)",
    )
    parser.add_argument(
        "--output_bucket",
        required=False,
        default="voxel-lightly-output",
        type=str,
        help="The bucket with the output of the lightly",
    )
    parser.add_argument(
        "--input_bucket",
        required=False,
        default="voxel-lightly-input",
        type=str,
        help="The bucket with the input of the lightly",
    )
    parser.add_argument(
        "--task",
        required=True,
        default="none",
        type=str,
        help="The task name to reference in output prefix",
    )
    parser.add_argument(
        "--remove-task-name",
        required=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to remove the task name from the start of the uuid (for example detector/camera_uuid/... becomes camera_uuid/...)",
    )
    parser.add_argument(
        "--use-epoch-time",
        required=False,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to use the epoch timestamp "
        "from the original video name (as with kinesis clips)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    lightly_token = json.loads(
        aws_utils.get_secret_from_aws_secret_manager(LIGHTLY_TOKEN_ARN)
    )["1"]
    lightly_client = ApiWorkflowClient(token=lightly_token)
    frame_sequence = output_frame_sequence(lightly_client, args.lightly_run_id)
    trim_lightly_clips(
        args.input_bucket,
        args.output_bucket,
        args.camera_uuid,
        frame_sequence,
        args.task,
        args.remove_task_name,
        args.use_epoch_time,
    ).resolve(SilentResolver())
