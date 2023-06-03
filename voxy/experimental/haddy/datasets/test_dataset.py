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
import os
import sys

import yaml

from core.datasets.generators.yolo import DatasetGeneratorYOLO
from core.infra.cloud.gcs_utils import dump_to_gcs
from core.structs.actor import ActorCategory

"""
Dataset Generation

Train:
Number of frames: 6800
Logset Name: data/logsets/common/train/2020-12-20

Val:
Number of frames: 860
Logset Name: data/logsets/common/val/2020-12-20

Test
Number of frames: 1700
Logset Name: data/logsets/common/test/2020-12-20
"""
ordered_classes_to_keep = [ActorCategory.PERSON]
pre_conversion_actor_filters = [lambda actor: actor.category in ordered_classes_to_keep]


def videos_from_logsets(file_path):
    videos = []
    with open(
        os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], file_path), "r"
    ) as f:
        for line in f:
            line = line.strip("\n")
            videos.append(line)
    return videos


if __name__ == "__main__":
    # TODO expose paths/dirs, classes_to_keep, ..., with argparse.
    dataset_output_path = "gs://voxel-users/harishma/tmp/temp_dataset"
    exp_output_path = "gs://voxel-users/harishma/tmp"
    dataset_generator = DatasetGeneratorYOLO(
        dataset_output_path, ordered_classes_to_keep, pre_conversion_actor_filters
    )

    videosets = []

    logset_name = "experimental/haddy/datasets/test_logset"
    videos = videos_from_logsets(logset_name)
    videosets.append(videos)

    for videoset in videosets:
        for video in videoset:
            print(video)
            dataset_generator.generate_for_video_uuid(video)
