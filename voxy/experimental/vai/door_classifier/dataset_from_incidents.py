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

train_UUIDS = [
    'americold/modesto/e_dock_north/ch12/open_door_invalid/b88e4521-e9a4-4e68-9f71-6b8178bae6df_video',
    'americold/modesto/e_dock_north/ch12/open_door_invalid/c30cf099-d0e2-484d-9db6-b9c4102797d3_video'
    'americold/modesto/e_dock_north/ch12/open_door_invalid/3b7b1c39-ccdf-4026-8aa6-6fd4608bcc69_video'
    'americold/modesto/e_dock_north/ch12/open_door_invalid/994a1430-0fd5-489e-8c0f-e7f0259d0d52_video'
]

test_UUIDS = [
   'americold/modesto/e_dock_north/ch12/open_door_invalid/71bff763-bc6f-467b-9634-cf0072a826db_video'
]

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
        "gs://voxel-users/vai/door_state_classification/dataset2"
    )

    # TRAIN
    for uuid in train_UUIDS:
        print(uuid)
        dataset_generator = DatasetGeneratorDoorClassification(
           base_dataset_output_path + "/train"
        )
        dataset_generator.generate_for_video_uuid(uuid)
