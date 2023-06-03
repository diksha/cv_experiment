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
import numpy as np
import os
from tqdm import tqdm
from core.infra.cloud.gcs_utils import upload_to_gcs
import cv2
from core.infra.cloud.gcs_utils import get_video_signed_url


def compute_background_image(video_uuid, num_frames_for_median=10):
    cap = cv2.VideoCapture(get_video_signed_url(video_uuid))

    # Randomly select frames
    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(
        size=num_frames_for_median
    )

    # Store selected frames in an array
    frames = []
    for fid in tqdm(frameIds):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        frames.append(frame)
    cap.release()

    # Calculate the median along the time axis
    print("\t    computing median...")
    median_rame = np.median(frames, axis=0).astype(dtype=np.uint8)
    print("\t    median computed")

    return median_rame


def remove_background(image, background_image):
    diff_3ch = cv2.absdiff(image, background_image)
    _, diff_3ch = cv2.threshold(diff_3ch, 25, 1, cv2.THRESH_BINARY)

    diff = np.max(diff_3ch, axis=2)
    diff_3ch = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)

    foreground_image = image * diff_3ch
    return foreground_image


def remove_background_TEST(video_uuid):

    path_prefix = os.path.join(
        os.path.expanduser("~"), "voxel", "experimental", "ramin", "debug"
    )
    exp_output_path = f"gs://voxel-users/ramin/debug/"

    print("computing bg...")
    median_frame = compute_background_image(video_uuid)

    video_id = video_uuid.replace("/", "-")
    path = os.path.join(path_prefix, f"{video_id}_background.jpg")

    cv2.imwrite(path, median_frame)
    upload_to_gcs(
        os.path.join(exp_output_path, f"{video_id}_background.jpg"), path, "image/jpeg"
    )

    cap = cv2.VideoCapture(get_video_signed_url(video_uuid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    video_path = os.path.join(path_prefix, f"{video_id}_bg_removed.avi")
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        10,
        (frame_width, frame_height),
    )

    # Loop over all frames
    ret = True
    k = 0
    print("Removing background...")
    while ret and k < 100:
        k += 1
        ret, frame = cap.read()
        if not ret:
            continue

        foreground_image = remove_background(frame, median_frame)
        out.write(foreground_image)

    # Release video object
    cap.release()
    out.release()
    upload_to_gcs(
        os.path.join(exp_output_path, f"{video_id}_bg_removed.avi"),
        video_path,
        "video/avi",
    )


def videos_from_logsets(file_path):
    videos = []
    with open(os.path.join(os.environ["REPO_PATH"], file_path), "r") as f:
        for line in f:
            line = line.strip("\n")
            videos.append(line)
    return videos


if __name__ == "__main__":
    val_logset_name = "data/logsets/ramin/hardhat/val_logset"
    val_videos = videos_from_logsets(val_logset_name)

    for video_uuid in val_videos:
        print(video_uuid)
        remove_background_TEST(video_uuid)
