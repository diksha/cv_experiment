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

from core.ml.data.extraction.label_generation.label_generator import (
    LabelGenerator,
)
from core.ml.data.extraction.utils.data_extraction_helpers import (
    generate_image_filename,
)
from core.structs.actor import ActorCategory
from core.structs.attributes import RectangleXCYCWH
from core.structs.frame import Frame
from core.utils.aws_utils import upload_fileobj_to_s3


class CocoLabelGenerator(LabelGenerator):
    def __init__(self, bbox_mode=1, is_crowd=False):
        self._label_id = 0
        self._bbox_mode = bbox_mode
        self._is_crowd = is_crowd
        self._coco_labels = {"images": [], "categories": [], "annotations": []}
        self._coco_labels["categories"] = [
            {
                "id": i,
                "name": actor.name,
            }
            for i, actor in enumerate(ActorCategory)
        ]
        self._content_type = "application/json"

    def process_frame(
        self,
        frame_timestamp_ms: int,
        frame: Frame,
    ) -> None:
        img_filename = generate_image_filename(frame_timestamp_ms)
        self._coco_labels["images"].append(
            {
                "id": frame_timestamp_ms,
                "file_name": img_filename,
                "height": frame.frame_height,
                "width": frame.frame_width,
            }
        )

        for actor in frame.actors:
            (x_center, y_center, width, height) = RectangleXCYCWH.from_polygon(
                actor.polygon
            ).to_list()

            self._coco_labels["annotations"].append(
                {
                    "id": self._label_id,
                    "image_id": frame_timestamp_ms,
                    "bbox": [
                        int(x_center - width / 2.0),
                        int(y_center - height / 2.0),
                        int(width),
                        int(height),
                    ],
                    "bbox_mode": self._bbox_mode,
                    "category_id": actor.category.value,
                    "iscrowd": self._is_crowd,
                    "area": width * height,
                }
            )

            self._label_id += 1

    def publish_labels(
        self,
        save_dir: str,
    ) -> None:
        file_name = "coco.json"
        save_path = f"s3://{save_dir}/{file_name}"
        label_data = json.dumps(self._coco_labels)
        upload_fileobj_to_s3(
            save_path,
            label_data.encode("utf-8"),
            content_type=self._content_type,
        )

    @property
    def label_type(self) -> str:
        return "coco"
