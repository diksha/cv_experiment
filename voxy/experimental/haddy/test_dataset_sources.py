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
from core.datasets.converters.dataset_sources.yolo import YoloDataset

classes_to_keep = ["BARE_HEAD", "BARE_CHEST", "HELMET", "HIGH_VISIBILITY_VEST", "PERSON"]
yolo_dataset_provider = YoloDataset(
            "gs://voxel-datasets/original/aimh.isti.cnr.it/dataset.yml",
            classes_to_keep=classes_to_keep,
            max_samples=100,
        )

for sample in yolo_dataset_provider.iter_samples():
    for label in sample:
        print(label)
        print("______________________________________")