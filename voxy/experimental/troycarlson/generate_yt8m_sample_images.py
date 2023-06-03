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
"""Captures sample frames as images from a directory of videos."""
import argparse
import datetime
import os

import cv2


def get_sample_frame_numbers(frame_count, fps):
    """Exclude first/last 15s; grab 5 frames from this inner range."""
    min_frame = int(fps * 15)
    max_frame = int(frame_count - (fps * 15))
    mid_frame = frame_count // 2
    quartered_frame_count = (max_frame - min_frame) // 4

    return (
        min_frame,
        min_frame + quartered_frame_count,  # 1/4-ish
        mid_frame,
        mid_frame + quartered_frame_count,  # 3/4-ish
        max_frame,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", default="/home/troycarlson/voxel_data/youtube8m", type=str
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Limit the number of source videos used during testing.",
    )
    return parser.parse_args()


def generate_images():
    count = 0
    args = parse_args()
    image_dir = os.path.join(args.directory, "images")

    for filename in os.listdir(args.directory):
        filepath = os.path.join(args.directory, filename)
        vcap = cv2.VideoCapture(filepath)
        fps = vcap.get(cv2.CAP_PROP_FPS)
        frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
        sample_frame_numbers = get_sample_frame_numbers(frame_count, fps)
        print("Processing frames for {}: {}".format(filename, sample_frame_numbers))

        for frame_number in sample_frame_numbers:
            try:
                vcap.set(1, frame_number)
                _, frame = vcap.read()
                image_path = "{}/{}_{}.jpg".format(image_dir, filename, frame_number)
                cv2.imwrite(image_path, frame)
            except BaseException:
                print("Failed to write frame: {} #{}".format(filename, frame_number))

        # Break early for testing
        count += 1
        if args.limit and count >= args.limit:
            break

    print("\nOutput directory:\n    {}".format(image_dir))


if __name__ == "__main__":
    generate_images()
