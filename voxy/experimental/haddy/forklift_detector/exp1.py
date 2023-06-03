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
"""
Tranforms:
None

Train:
Number of frames: 6800
Logset Name: data/logsets/haddy/americold-2020-11-28-train

Val:
Number of frames: 860
Logset Name: data/logsets/haddy/americold-2020-11-28-val

Test
Number of frames: 1700
Logset Name: data/logsets/haddy/americold-2020-11-28-test
"""
import os
import sys

import yaml

from core.datasets.generators.yolo import DatasetGeneratorYOLO
from core.infra.cloud.gcs_utils import dump_to_gcs
from core.structs.actor import ActorCategory


def videos_from_logsets(file_path):
    videos = []
    with open(
        os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], file_path), "r"
    ) as f:
        for line in f:
            line = line.strip("\n")
            videos.append(line)
    return videos


training_logset_name = "data/logsets/haddy/americold-2020-11-28-train"
train_videos = videos_from_logsets(training_logset_name)

val_logset_name = "data/logsets/haddy/americold-2020-11-28-val"
val_videos = videos_from_logsets(val_logset_name)

exp_output_path = "gs://voxel-users/haddy/yolo/dataset1"

# YOLO dataset file
dataset_dict = {
    "train": [f"/data/{train_video}/images" for train_video in train_videos],
    "val": [f"/data/{val_video}/images" for val_video in val_videos],
    "nc": 1,
    "names": ["PIT"],
}
dataset_yaml = yaml.dump(dataset_dict)

dump_to_gcs(os.path.join(exp_output_path, "dataset1.yaml"), dataset_yaml, "text/plain")
