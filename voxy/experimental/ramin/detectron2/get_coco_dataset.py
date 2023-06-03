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
os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://root:passwordA1@34.127.56.54:27017/admin"


import fiftyone as fo
import fiftyone.zoo as foz

data_dir = '~/data/fiftyone'
data_dir = os.path.expanduser(data_dir)
coco_person_dir = os.path.join(data_dir, 'coco_person')

os.makedirs(data_dir, exist_ok=True)
os.makedirs(coco_person_dir, exist_ok=True)

fo.config.default_dataset_dir = data_dir
fo.config.dataset_zoo_dir = data_dir
print(fo.config)

dataset = foz.load_zoo_dataset(
   "coco-2017",
   split="train",
   dataset_name="persons_train",
   classes=("person"),
)

coco_person_10_dir = os.path.join(data_dir, 'coco_person_10k')
os.makedirs(coco_person_10_dir, exist_ok=True)

dataset.take(10000).export(
    export_dir=coco_person_10_dir,
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
    labels_path=os.path.join(coco_person_10_dir, "coco_labels.json"),
)

import os
import json

json_file = os.path.join(coco_person_10_dir, "coco_labels.json")
with open(json_file) as f:
    dataset_dicts = json.load(f)


dataset_dicts['categories'] = [{'id': 1, 'name': 'PERSON'},
 {'id': 2, 'name': 'PIT'},
 {'id': 3, 'name': 'HARDHAT'},
 {'id': 4, 'name': 'SAFETY_VEST'}]

annotations = []
for anno in dataset_dicts['annotations']:
    if anno['category_id'] == 1:
        annotations.append(anno)
        
dataset_dicts['annotations'] = annotations

with open(os.path.join(coco_person_10_dir, "coco_labels_filtered.json"), 'w') as fp:
    json.dump(dataset_dicts, fp)

os.rename(os.path.joint(coco_person_10_dir, "data"), os.path.joint(coco_person_10_dir, "images"))