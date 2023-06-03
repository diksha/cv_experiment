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

# Experimental tool I wrote to fix the timestamps of some segmented videos which were segmented using ffmpeg.
import os
from core.infra.cloud import gcs_utils

VIDEO_FORMAT = "mp4"

if __name__ == "__main__":
    video_dirs = [
        "/home/haddy/voxel/data/americold/modesto/e_dock_north/ch04_1",
        "/home/haddy/voxel/data/uscold/laredo/dock03/cha/",
        "/home/haddy/voxel/data/uscold/laredo/doors_14_20/cha",
        "/home/haddy/voxel/data/uscold/laredo/doors_19_27/cha/",
        "/home/haddy/voxel/data/uscold/laredo/room_f1/cha/",
    ]

    ffmpeg_command_template = "ffmpeg -i {} -y -c copy -map 0 -reset_timestamps -o {}"
    failed_conversion = set()
    for video_dir in video_dirs:
        print("Processing", video_dir)
        for f in os.listdir(video_dir):
            if f.endswith(VIDEO_FORMAT):
                local_video_path = os.path.join(video_dir, f)
                print("Processing ", local_video_path)
                ffmpeg_command = ffmpeg_command_template.format(
                    local_video_path, local_video_path
                )
                try:
                    ret = os.system(ffmpeg_command)
                    if ret != 0:
                        failed_conversion.append(local_video_path)
                except:
                    failed_conversion.append(local_video_path)

    if len(failed_conversion) > 0:
        print("Failed to convert the following. Please try again.")
        for f in failed_conversion:
            print(f)
