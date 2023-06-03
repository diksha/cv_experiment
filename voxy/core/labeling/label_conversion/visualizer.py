#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

# Example Usage: ./bazel run core/labeling/label_conversion:visualizer -- --datacollection_uuid
# wesco/reno/0002/cha/2b0d35d2-85f0-4695-81be-0768efd95d1f --file_path
# /home/diksha/file.mcap

import argparse
import json

from loguru import logger
from mcap_protobuf.writer import Writer

from core.labeling.label_store.label_reader import LabelReader
from core.structs.frame import Frame, StampedImage
from core.utils.aws_utils import (
    does_blob_exist,
    download_cv2_imageobj_to_memory,
)
from core.utils.video_reader import S3VideoReader, S3VideoReaderInput


def log_labels(mcap_writer: Writer, frame: dict, timestamp: int):
    """Log labels for a single frame

    Args:
        mcap_writer (Writer): mcap writer
        frame (dict): Labels for a frame
        timestamp (int): timestamp to log
    """
    frame_label = Frame.from_dict(frame)
    base_topic = "/Label"
    mcap_writer.write_message(
        topic=base_topic,
        message=frame_label.to_proto(),
        log_time=timestamp,
        publish_time=timestamp,
    )
    base_topic = "/".join(["", "Label", "annotations"])
    mcap_writer.write_message(
        topic=base_topic,
        message=frame_label.to_annotation_protos(),
        log_time=timestamp,
        publish_time=timestamp,
    )


def log_image_collection(
    mcap_writer: Writer, datacollection_uuid: str, labels: dict
):
    """Log image collection

    Args:
        mcap_writer (Writer): mcap writer
        datacollection_uuid (str): uuid of the image data collection
        labels (dict): dict of labels
    """
    logger.info("logging image data collection")
    timestamp_s = 0
    bucket_name = "voxel-logs"
    for frame in labels["frames"]:
        timestamp_ns = int(timestamp_s * 1e9)
        image_key = f"{datacollection_uuid}/{frame['relative_image_path']}"
        frame_image = download_cv2_imageobj_to_memory(bucket_name, image_key)

        # Log images
        stamped_image = StampedImage(
            image=frame_image, timestamp_ms=timestamp_s * 1000
        )
        base_topic = "/".join(["", "Label", stamped_image.get_topic_name()])
        mcap_writer.write_message(
            topic=base_topic,
            message=stamped_image.to_proto(),
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
        )

        # Log labels
        log_labels(mcap_writer, frame, timestamp_ns)
        timestamp_s = timestamp_s + 1


def log_video(mcap_writer: Writer, video_uuid: str, labels: dict):
    """Log video data to mcap

    Args:
        mcap_writer (Writer): mcap writer
        video_uuid (str): uuid of the video
        labels (dict): Labels for a frame
    """

    logger.info("Logging video labels")
    frames_tms_to_extract = {
        frame["relative_timestamp_ms"] for frame in labels["frames"]
    }
    ts_to_fram_map = {}
    for frame in labels["frames"]:
        ts_to_fram_map[frame["relative_timestamp_ms"]] = frame
    video_reader = S3VideoReader(
        S3VideoReaderInput(video_path_without_extension=video_uuid)
    )
    for video_reader_op in video_reader.read():
        frame_ms = video_reader_op.relative_timestamp_ms
        frame = video_reader_op.image
        if frame_ms in frames_tms_to_extract:
            log_time = int(frame_ms * 1e6)
            stamped_image = StampedImage(image=frame, timestamp_ms=frame_ms)
            base_topic = "/".join(
                ["", "Label", stamped_image.get_topic_name()]
            )
            mcap_writer.write_message(
                topic=base_topic,
                message=stamped_image.to_proto(),
                log_time=log_time,
                publish_time=log_time,
            )
            log_labels(mcap_writer, ts_to_fram_map[frame_ms], log_time)


def log_data_collection(datacollection_uuid: str, file_path: str):
    """Log data collection to mcap

    Args:
        datacollection_uuid (str): data collection uuid
        file_path (str): file path to log mcap
    """
    video_path = f"s3://voxel-logs/{datacollection_uuid}.mp4"
    labels = json.loads(LabelReader().read(datacollection_uuid))
    with open(file_path, "wb") as out_file, Writer(out_file) as mcap_writer:
        if does_blob_exist(video_path):
            log_video(mcap_writer, datacollection_uuid, labels)
        else:
            log_image_collection(mcap_writer, datacollection_uuid, labels)
    logger.info(f"See mcap logs at {file_path}")


def parse_args() -> argparse.Namespace:
    """Parse arguments
    Returns:
        argparse.Namespace: argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datacollection_uuid",
        "-d",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--file_path",
        "-f",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_data_collection(args.datacollection_uuid, args.file_path)
