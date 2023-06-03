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

# This is used to import anything that needs to be registered
#
# Unused imports:
# trunk-ignore-all(pylint/W0611)
# trunk-ignore-all(flake8/F401)


# Converters:
import core.labeling.scale.lib.converters.image_collection_annotation_converter
import core.labeling.scale.lib.converters.video_playback_annotation_converter

# Task Creators:
import core.labeling.scale.task_creation.door_classification_task
import core.labeling.scale.task_creation.image_segmentation_task
import core.labeling.scale.task_creation.ppe_hat_image_annotation_task
import core.labeling.scale.task_creation.safety_gloves_image_annotation_task
import core.labeling.scale.task_creation.safety_vest_image_annotation_task
import core.labeling.scale.task_creation.video_playback_annotation_task
