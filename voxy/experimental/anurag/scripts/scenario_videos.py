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
import json
import os

from google.cloud import storage


def copy_blob(bucket_name, blob_name, new_bucket_name, new_blob_name):
    storage_client = storage.Client("sodium-carving-227300")
    source_bucket = storage_client.get_bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.get_bucket(new_bucket_name)
    source_bucket.copy_blob(source_blob, destination_bucket, new_blob_name)


def main():
    filename = os.path.join(
        os.environ["BUILD_WORKSPACE_DIRECTORY"],
        "data",
        "scenario_sets",
        "americold",
        "modesto",
        "first.json",
    )
    scenarios = json.loads(open(filename).read())

    for scenario in scenarios["scenarios"]:
        if scenario["video_uuid"].startswith("voxel-portal"):
            blob_uuid = scenario["video_uuid"].replace("voxel-portal/", "")
            source_blob_name = f"{blob_uuid}.mp4"
            camera_uuid = scenario["camera_uuid"]
            incident_uuid = (
                scenario["video_uuid"]
                .replace("voxel-portal/incidents/", "")
                .replace("_video", "")
            )
            destination_blob_name = (
                f"{camera_uuid}/scenarios/{incident_uuid}.mp4"
            )
            copy_blob(
                "voxel-portal",
                source_blob_name,
                "voxel-logs",
                destination_blob_name,
            )
            scenario["video_uuid"] = destination_blob_name.replace(".mp4", "")

    with open(filename, "w") as f:
        json.dump(scenarios, f, indent=4)


main()
