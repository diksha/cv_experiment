#
# Copyright 2022-2023 Voxel Labs, Inc.
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
import sys
from typing import List, Optional

from core.infra.sematic.perception.data_ingestion.ingest_raw_videos.pipeline import (
    pipeline,
)
from core.infra.sematic.shared.utils import (
    PipelineSetup,
    SematicOptions,
    resolve_sematic_future,
)
from core.utils.aws_utils import (
    does_s3_blob_exists,
    read_decoded_bytes_from_s3,
)

INPUT_BUCKET = "voxel-raw-logs"


def aggregate_video_uuids(
    videos: List[str], s3_file_path: Optional[str]
) -> List[str]:
    """Support for chunking and ingesting multiple vidoes in voxel-raw-logs using local_file_path
    Args:
        videos (List[str]): manually input videos
        s3_file_path (Optional[str]): optional s3 file path containing line delimited video_uuids
            for ingestion
    Returns:
        List[str]: list of video uuids to ingest
    Raises:
        RuntimeError: local_file_path specified but does not exist
        RuntimeError: video list is empty
    """
    if "NONE" in list(map(lambda x: x.upper(), videos)):
        idx = list(map(lambda x: x.upper(), videos)).index("NONE")
        del videos[idx]
    if s3_file_path:
        if not does_s3_blob_exists(s3_file_path):
            raise RuntimeError(
                f"Local file path specified but does not exist, {s3_file_path}"
            )
        s3_videos = read_decoded_bytes_from_s3(s3_file_path).split("\n")
        videos.extend(s3_videos)
    if len(videos) < 1:
        raise RuntimeError("No video uuids provided")
    for video in videos:
        s3_video_path = f"s3://{INPUT_BUCKET}/{video}.mp4"
        if not does_s3_blob_exists(s3_video_path):
            raise RuntimeError(f"Video does not exist, {s3_video_path}")
    return videos


def main(
    videos: List[str],
    is_test: bool,
    fps: float,
    sematic_options: SematicOptions,
) -> int:
    """Ingest videos to scale

    Args:
        videos: See command line help
        is_test: See command line help
        fps (float): See command line help
        sematic_options (SematicOptions): options for Sematic resolver

    Returns:
        The return code that should be used for the process
    """
    # trunk-ignore(pylint/E1101)
    future = pipeline(
        videos=videos,
        is_test=is_test,
        fps=fps,
        metaverse_environment=os.environ.get(
            "METAVERSE_ENVIRONMENT", "INTERNAL"
        ),
        pipeline_setup=PipelineSetup(),
    ).set(name="Ingest raw videos", tags=["P3"])
    resolve_sematic_future(future, sematic_options)
    return 0


def parse_args() -> argparse.Namespace:
    """Parses arguments for the binary

    Returns:
        argparse.Namespace: Command line arguments passed in.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos",
        type=str,
        nargs="+",
        default=[],
        help="Video uuids from voxel-raw-logs to ingest",
    )
    parser.add_argument(
        "--s3_file_path",
        type=str,
        default="None",
        help=(
            "Another way to ingest video uuids in voxel-raw-logs. s3 file "
            "path points to file in s3 containing line delimited video "
            "uuids to ingest"
        ),
    )
    parser.add_argument(
        "--is_test",
        type=str,
        default="false",
        help=(
            "Boolean string used to mark whether or not a video is used "
            "for test"
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help=(
            "FPS used for videos sent to labeling. FPS < 0 means video is "
            "not sent for labeling. FPS == 0.0 means use original video FPS"
        ),
    )
    SematicOptions.add_to_parser(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vids = aggregate_video_uuids(
        args.videos,
        args.s3_file_path if args.s3_file_path.upper() != "NONE" else None,
    )
    sys.exit(
        main(
            vids,
            args.is_test.upper() == "TRUE",
            args.fps,
            SematicOptions.from_args(args),
        )
    )
