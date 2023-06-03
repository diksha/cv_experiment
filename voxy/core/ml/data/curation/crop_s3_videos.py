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
import os
import tempfile
import typing

import av
import sematic
from loguru import logger
from sematic.resolvers.silent_resolver import SilentResolver
from tqdm import tqdm

from core.incidents.utils import CameraConfig
from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.ml.data.curation.crop_video import crop_video
from core.structs.attributes import RectangleXYWH
from core.utils import aws_utils


def get_door_info_from_video(
    camera_uuid: str, video_path: str, index: int
) -> typing.Tuple[RectangleXYWH, int]:
    """
    Grab door rectangle x, y, w, h from the uuid and the local path for the video along with the
    door id

    Args:
        camera_uuid (str): the camera uuid
        video_path (str): the video path (local). This is needed to get the width and height
        index (int): the index of the door to grab a rectangle for

    Returns:
        typing.Tuple[RectangleXYWH, int]: the rectangle for the door, door id
    """
    # get width , height
    container = av.open(video_path)
    width = container.streams.video[0].width
    height = container.streams.video[0].height
    # TODO make this work for multiple doors
    camera_config = CameraConfig(camera_uuid, height, width)
    door_polygon = camera_config.doors[index].polygon
    door_id = camera_config.doors[index].door_id
    return RectangleXYWH.from_polygon(door_polygon), door_id


def get_number_of_doors(camera_uuid: str) -> int:
    return len(CameraConfig(camera_uuid, 1, 1).doors)


def crop_door_s3_video(
    input_bucket: str,
    input_video_path: str,
    output_bucket: str,
    output_video_path_format_str: str,
    camera_uuid: str,
):
    """
    Crops the door from a s3 video and writes to the output path

    Args:
        input_bucket (str): the input bucket to read videos from
        input_video_path (str): the path in the bucket for the video: s3://input_bucket/input_video_path
        output_bucket (str): the output bucket for the video
        output_video_path_format_str (str):  the output video path format string. The cropped video will be written to: s3://output_bucket/output_video_path
                                             formatted by the door id
        camera_uuid (str): the camera uuid to get the door config from
    """
    with tempfile.NamedTemporaryFile(
        suffix=".mp4"
    ) as tmp_input_video, tempfile.NamedTemporaryFile(
        suffix=".mp4"
    ) as tmp_output_video:
        # download input video path
        aws_utils.download_to_file(
            input_bucket, input_video_path, tmp_input_video.name
        )
        n_doors = get_number_of_doors(camera_uuid)
        for i in range(n_doors):
            door_rectangle, door_id = get_door_info_from_video(
                camera_uuid, tmp_input_video.name, i
            )
            crop_video(
                tmp_input_video.name, tmp_output_video.name, door_rectangle
            )
            output_video_path = output_video_path_format_str.format(door_id)
            # upload to s3
            aws_utils.upload_file(
                output_bucket, tmp_output_video.name, output_video_path
            )


# trunk-ignore-begin(pylint/W9015,pylint/W9011,pylint/W9006)
@sematic.func(
    resource_requirements=CPU_1CORE_4GB,
    standalone=True,
)
def crop_all_doors_from_videos(
    input_bucket: str,
    video_path: str,
    output_bucket: str,
    project: str,
    camera_uuid: str,
    extension: str = "mp4",
    input_prefix_to_remove: typing.Optional[str] = None,
) -> str:
    """# Crops all doors into new videos stored in S3.

    The input videos are stored in:
    s3://{input_bucket}/{video_path}*.{extension}

    The cropped videos are stored in:
    s3://{output_bucket}/{project}/{video_path}*.{extension}

    ## Parameters
    - **input_bucket**:
        The bucket to pull videos from
    - **video_path**:
        The path within the input bucket to pull videos from
    - **output_bucket**:
        The bucket to write cropped videos into
    - **project**:
        The project to store the cropped video
    - **camera_uuid**:
        The UUID of the camera the video is from
    - **extension**:
        The extension to glob. Defaults to "mp4".
    - **input_prefix_to_remove**:
        Optional prefix we want to try to remove from the input video path
        Should only be used for door data ingestion pipelines.

    ## Returns
    The s3 URI for the "directory" where the cropped videos were written

    ## Raises
    **RuntimeError**:  if there are no videos to crop from the input bucket
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011,pylint/W9006)
    videos = aws_utils.glob_from_bucket(input_bucket, video_path, (extension))
    if not videos:
        raise RuntimeError(
            f"No videos were found for s3://{input_bucket}/{video_path}/*.mp4"
        )

    for video in tqdm(videos):
        logger.info(f"Processing {video}")
        video_name, extension = os.path.splitext(video)
        if input_prefix_to_remove is not None:
            common_prefix = os.path.commonprefix(
                [video_name, input_prefix_to_remove]
            )
            if common_prefix == input_prefix_to_remove:
                video_name = os.path.relpath(
                    video_name, input_prefix_to_remove
                )
        path_name = os.path.join(project, video_name)
        output_path_format_string = "".join(
            (f"{path_name}", "_door_{}", f"{extension}")
        )

        crop_door_s3_video(
            input_bucket,
            video,
            output_bucket,
            output_path_format_string,
            camera_uuid,
        )
    return f"s3://{output_bucket}/{project}"


def get_args() -> None:
    parser = argparse.ArgumentParser(description="Convert camera config")
    parser.add_argument(
        "--camera-uuid",
        type=str,
        required=True,
        help="The camera uuid ",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=False,
        default="doors/cropped",
        help="Project to output the crops to",
    )
    parser.add_argument(
        "--output-bucket",
        type=str,
        required=False,
        default="voxel-lightly-input",
        help="The bucket to write the to.",
    )
    parser.add_argument(
        "--input-bucket",
        type=str,
        required=False,
        default="voxel-lightly-input",
        help="The bucket to read from.",
    )
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()
    crop_all_doors_from_videos(
        args.input_bucket,
        args.camera_uuid,
        args.output_bucket,
        args.project,
        args.camera_uuid,
    ).resolve(SilentResolver())
