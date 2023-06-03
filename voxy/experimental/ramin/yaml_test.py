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
import yaml

dummy_videos = ["one.mp4", "two.mp4", "three.mp4"]

dataset_dict = {
    "train": [f"/data/{train_video}/images" for train_video in dummy_videos],
    "val": [f"/data/{val_video}/images" for val_video in dummy_videos],
    "test": [f"/data/{test_video}/images" for test_video in dummy_videos],
    "nc": 3,
    "names": ["HARDHAT", "PERSON", "SAFETY_VEST"],
}
dataset_yaml = yaml.dump(dataset_dict)

print(dataset_yaml)
