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
import sys

import os
paths_to_remove = []
for path in sys.path:
   if "pip_deps_pypi__fiftyone_brain" in path or "pip_deps_pypi__fiftyone_db" in path:
      paths_to_remove.append(path)
for path in paths_to_remove:
   sys.path.remove(path)
os.environ['PATH'] = f"{os.environ['PATH']}:external/pip_deps_pypi__fiftyone_db/fiftyone/db/bin/"
import fiftyone as fo
import fiftyone.zoo as foz
print("Hello")
dataset = foz.load_zoo_dataset(
   "coco-2017",
   split="validation",
   dataset_name="persons",
   classes=("person"),
   max_num_samples=2
)
# dataset.persistent = True
# samples = dataset.take(2, seed=51)
# for sample in samples:
#  print(sample.filepath)
#  # print(sample.field_names)
#  print(sample.ground_truth)  # this includes bbox for each person in the image