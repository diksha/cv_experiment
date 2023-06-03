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
from core.labeling.scale.lib.converters.door_actor_utils import (
    generate_consumable_labels_for_doors,
)
from core.labeling.scale.lib.converters.image_collection_annotation_converter import (
    ImageCollectionAnnotationConverter,
)
from core.labeling.scale.lib.converters.video_playback_annotation_converter import (
    VideoPlaybackAnnotationConverter,
)
from core.labeling.scale.task_creation.door_classification_task import (
    DoorClassificationTask,
)
from core.labeling.scale.task_creation.image_segmentation_task import (
    ImageSegmentationTask,
)
from core.labeling.scale.task_creation.video_playback_annotation_task import (
    VideoPlaybackAnnotationTask,
)

task_metadata = {
    "videoplaybackannotation": {
        "creation_fn": VideoPlaybackAnnotationTask,
        "converter_fn": VideoPlaybackAnnotationConverter,
        "project_name": "video_playback_annotation",
        "consumable_labels_fn": None,
    },
    "doorclassification": {
        "creation_fn": DoorClassificationTask,
        "converter_fn": ImageCollectionAnnotationConverter,
        "project_name": "door_state_classification",
        "consumable_labels_fn": generate_consumable_labels_for_doors,
    },
    "imagesegmentation": {
        "creation_fn": ImageSegmentationTask,
        "converter_fn": None,
        "project_name": "image_segmentation_prod",
        "consumable_labels_fn": None,
    },
}
