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

from core.ml.data.extraction.label_generation.label_generator import (
    LabelGenerator,
)
from core.ml.data.extraction.utils.data_extraction_helpers import (
    get_all_actor_classes,
)
from core.structs.actor import ActorCategory
from core.structs.attributes import RectangleXCYCWH
from core.structs.frame import Frame
from core.utils.aws_utils import upload_fileobj_to_s3


class YoloLabelGenerator(LabelGenerator):
    def __init__(self, actors_to_keep: list = None):
        self._label_id = 0
        self._yolo_labels = {}
        self._content_type = "text/plain"
        self._yolo_detector_classes = get_all_actor_classes()
        if actors_to_keep is not None:
            self._yolo_detector_classes = actors_to_keep

    def _convert_actor_class_to_yolo_class(
        self,
        actor_category: ActorCategory,
    ) -> int:
        yolo_class = -1
        if actor_category in self._yolo_detector_classes:
            yolo_class = self._yolo_detector_classes.index(actor_category)
        return yolo_class

    def process_frame(
        self,
        frame_timestamp_ms: int,
        frame: Frame,
    ) -> None:
        label_output = []
        for actor in frame.actors:
            if actor.category not in self._yolo_detector_classes:
                continue
            (x_center, y_center, width, height) = RectangleXCYCWH.from_polygon(
                actor.polygon
            ).to_list()

            yolo_class = self._convert_actor_class_to_yolo_class(
                actor.category
            )
            norm_xc = float(x_center) / frame.frame_width
            norm_yc = float(y_center) / frame.frame_height
            norm_w = float(width) / frame.frame_width
            norm_h = float(height) / frame.frame_height
            label_output.append(
                f"{yolo_class} {norm_xc} {norm_yc} {norm_w} {norm_h}"
            )
        self._yolo_labels[frame_timestamp_ms] = "\n".join(label_output)

    def publish_labels(
        self,
        save_dir: str,
    ) -> None:
        for ts, frame_label_data in self._yolo_labels.items():
            save_path = f"s3://{save_dir}/frame_{ts}.txt"
            upload_fileobj_to_s3(
                save_path,
                frame_label_data.encode("utf-8"),
                content_type=self._content_type,
            )

    @property
    def label_type(self) -> str:
        return "yolo"
