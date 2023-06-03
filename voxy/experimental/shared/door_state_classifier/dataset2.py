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

from core.datasets.generators.door_classification_from_incidents import (
    DatasetGeneratorDoorClassification,
)
from core.infra.cloud.gcs_utils import dump_to_gcs
from core.structs.actor import ActorCategory

"""
Dataset Generation

Train:
Logset Name: data/logsets/common/door_state_classification/train-2021-04-19

Val:
Logset Name: data/logsets/common/door_state_classification/val-2021-04-19

Test
Logset Name: data/logsets/common/door_state_classification/test-2021-04-19

for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset1/train/closed);
do echo TRAIN,$f,CLOSED;
done >> labels.csv;

for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset1/train/open);
do echo TRAIN,$f,OPEN;
done >> labels.csv;

for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset1/val/closed);
do echo VALIDATION,$f,CLOSED;
done >> labels.csv;

for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset1/val/open);
do echo VALIDATION,$f,OPEN;
done >> labels.csv;

for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset1/test/closed);
do echo TEST,$f,CLOSED;
done >> labels.csv;


for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset1/test/open);
do echo TEST,$f,OPEN;
done >> labels.csv;

for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset2/train/closed);
do echo TRAIN,$f,CLOSED;
done >> labels.csv;

for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset2/train/open);
do echo TRAIN,$f,OPEN;
done >> labels.csv;

for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset2/val/closed);
do echo VALIDATION,$f,CLOSED;
done >> labels.csv;

for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset2/val/open);
do echo VALIDATION,$f,OPEN;
done >> labels.csv;

for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset2/test/closed);
do echo TEST,$f,CLOSED;
done >> labels.csv;


for f in $(gsutil ls gs://voxel-ml-datasets/common/door_state_classification/dataset2/test/open);
do echo TEST,$f,OPEN;
done >> labels.csv;

"""


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
    base_dataset_output_path = (
        "gs://voxel-users/common/door_state_classification/dataset2"
    )

    # TRAIN
    training_logset_name = (
        "data/logsets/common/door_state_classification/train-2021-04-19"
    )
    for video in videos_from_logsets(training_logset_name):
        print(video)
        dataset_generator = DatasetGeneratorDoorClassification(
            f"{base_dataset_output_path}/train"
        )
        dataset_generator.generate_for_video_uuid(video)

    # VAL
    val_logset_name = "data/logsets/common/door_state_classification/val-2021-04-19"
    for video in videos_from_logsets(val_logset_name):
        print(video)
        dataset_generator = DatasetGeneratorDoorClassification(
            f"{base_dataset_output_path}/val"
        )
        dataset_generator.generate_for_video_uuid(video)

    # TEST
    test_logset_name = "data/logsets/common/door_state_classification/test-2021-04-19"
    for video in videos_from_logsets(test_logset_name):
        print(video)
        dataset_generator = DatasetGeneratorDoorClassification(
            f"{base_dataset_output_path}/test"
        )
        dataset_generator.generate_for_video_uuid(video)
