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

import json
import os
import tempfile
from collections import defaultdict

from core.infra.cloud.gcs_utils import (
    download_blob,
    dump_to_gcs,
    get_storage_client,
)


def download_labels(
    storage_client,
    logset_path: str = "/home/nasha_voxelsafety_com/voxel/data/logsets/nasha/ppe_classification/all-2022-03-28",
    gcs_path: str = "gs://voxel-users/nasha/ppe_detection/detr/",
) -> list:
    with open(
        logset_path,
        "r",
        encoding="UTF-8",
    ) as logset:
        video_uuids = logset.readlines()
    annots = []
    for video_uuid in video_uuids:
        video_uuid_flattend = video_uuid.strip().replace("/", "_")
        print(os.path.join(gcs_path, video_uuid_flattend, "coco_labels.json"))
        with tempfile.NamedTemporaryFile(suffix=".json") as temp:
            download_blob(
                os.path.join(
                    gcs_path, video_uuid_flattend, "coco_labels.json"
                ),
                temp.name,
                storage_client=storage_client,
            )
            annot_dict = json.load(temp)
            annots.append(annot_dict)
    return annots


def merge_labels(labels: list) -> None:

    dd = defaultdict(list)
    dd["categories"] = labels[0]["categories"]
    for d in labels:  # you can list as many input dicts as you want here
        for key, value in d.items():
            if key != "categories":
                dd[key].extend(value)

    map_ids = {}
    for i, img in enumerate(dd["images"]):
        video_uuid = img["id"]
        if map_ids.get(video_uuid) is None:
            map_ids[video_uuid] = i
        else:
            print(video_uuid)
        dd["images"][i]["id"] = map_ids[video_uuid]

    for i, annot in enumerate(dd["annotations"]):
        video_uuid = annot["image_id"]
        dd["annotations"][i]["image_id"] = map_ids[video_uuid]

    return dict(dd)


if __name__ == "__main__":
    storage_client = get_storage_client()
    labels = download_labels(storage_client=storage_client)
    label_dict = merge_labels(labels=labels)
    with open(
        "/home/nasha_voxelsafety_com/voxel/experimental/nasha/data/coco_labels.json",
        "w",
    ) as outfile:
        json.dump(label_dict, outfile)
