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
ordered_classes_to_keep = [ActorCategory.PIT]
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

    # TRAIN
    training_logset_name = "experimental/haddy/test_logset"
    train_videos = videos_from_logsets(training_logset_name)
    videosets.append(train_videos)

    # # VAL
    # val_logset_name = "data/logsets/common/val/2020-12-20"
    # val_videos = videos_from_logsets(val_logset_name)
    # videosets.append(val_videos)

    # # TEST
    # test_logset_name = "data/logsets/common/test/2020-12-20"
    # test_videos = videos_from_logsets(test_logset_name)
    # videosets.append(test_videos)

    for videoset in videosets:
        for video in videoset:
            print(video)
            dataset_generator.generate_for_video_uuid(video)

    # YOLO dataset file
    # dataset_dict = {
    #     "train": [f"/data/{train_video}/images" for train_video in train_videos],
    #     "val": [f"/data/{val_video}/images" for val_video in val_videos],
    #     "test": [f"/data/{test_video}/images" for test_video in test_videos],
    #     "nc": 1,
    #     "names": ["PIT"],
    # }
    # dataset_yaml = yaml.dump(dataset_dict)

    # dump_to_gcs(
    #     os.path.join(exp_output_path, "dataset1.yaml"), dataset_yaml, "text/plain"
    # )
