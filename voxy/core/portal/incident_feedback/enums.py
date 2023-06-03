#
# Copyright 2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
from enum import Enum


class IncidentFeedbackType(Enum):
    ACCURACY = "incident_accuracy"


class IncidentAccuracyOption(Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNSURE = "unsure"
    CORRUPT = "corrupt"
