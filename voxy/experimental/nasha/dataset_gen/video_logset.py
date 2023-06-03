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
from core.metaverse.metaverse import Metaverse
import os
from core.infra.symphony.api.firestore import append_to_symphony_collection
import argparse 
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logset_name", "-l", type=str, required=True)
    return parser.parse_args()

def main():
    metaverse = Metaverse(environment="PROD")
    args = parse_args()
    result = metaverse.schema.execute(f'{{ logset(name:"{args.logset_name}"){{   name, video_ref {{ video {{uuid}} , violation_version, labels_version }} }} }}')
    print("Result of graphql query to get logsets : ", result)
    result = result.data
    if result is not None:
        videos = []
        for logset in result["logset"]:
            for video in logset["video_ref"]:
                videos.append(json.dumps(video, separators=(',', ':')))

        if os.getenv("SYMPHONY_CONFIG_FIRESTORE_UUID"):
            append_to_symphony_collection(
                os.getenv("SYMPHONY_CONFIG_FIRESTORE_UUID"),
                "logset_videos",
                videos,
            )
        else:
            raise RuntimeError("SYMPHONY_CONFIG_FIRESTORE_UUID not set")


if __name__ == "__main__":
    main()