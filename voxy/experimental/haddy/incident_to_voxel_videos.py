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
import subprocess
import sys


def generate_commands_to_upload_videos_as_voxel_videos(csv_file_path):
    commands = []
    with open(csv_file_path, "r") as f:
        i = -1
        for line in f:
            i = i + 1
            if i == 0:
                continue
            columns = line.split(",")
            camera_uuid = columns[2].strip("\n")
            video_path = columns[3].strip("\n")
            commands.append(
                f"gsutil cp {video_path} gs://voxel-logs/{camera_uuid}/open_door_invalid/"
            )
    f.close()
    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incident to Voxel Videos Utils"
    )
    parser.add_argument(
        "csv_file_path",
        help="Path to a CSV file containing paths of videos to upload and other useful metadata. Look at \
        https://docs.google.com/spreadsheets/d/1XND4aT996xVi_FsFoa3_3jm3FBPbxKvU75BTxS6UaHs/edit?usp=sharing \
        for an sample format.",
    )
    args = parser.parse_args()
    commands = generate_commands_to_upload_videos_as_voxel_videos(
        args.csv_file_path
    )
    failed_command = []
    for command in commands:
        result = subprocess.run(command.split(" "))
        if result.returncode != 0:
            failed_command.append(command)
        else:
            print(command, "successful")

    print("The following command failed:", failed_command)
