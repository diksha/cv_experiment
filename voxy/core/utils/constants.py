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

# floating point constants
FLOATING_POINT_ERROR_THRESHOLD = 0.001

# Filter Input Threshold
CONFIDENCE_SCORE_THRESHOLD = 0.65
FILTER_THRESHOLD_FOR_LOW_CONFIDENCE_KEYPOINTS = 5
NECK_ANGLE_THRESHOLD = 20
LEGS_INFRONT_OF_EACH_OTHER_SLOPE_THRESHOLD = 1

# Ergonomics should always assume full body is visible.
# TODO(harishma): This might not be relevant in warehouse setting.
FULL_VIEW_MODE_ON = True
