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
import typing
from enum import Enum

import attr


class ActivityType(Enum):
    UNKNOWN = 0
    LIFTING = 1
    REACHING = 2


class PostureType(Enum):
    BAD = 0
    GOOD = 1
    UNKNOWN = 2


@attr.s(slots=True)
class Activity:

    activity: typing.Optional[ActivityType] = attr.ib(default=None)
    posture: typing.Optional[PostureType] = attr.ib(default=None)

    def to_dict(self):
        return {
            "activity": self.activity.name
            if self.activity is not None
            else None,
            "posture": self.posture.name if self.posture is not None else None,
        }

    @classmethod
    def from_dict(cls, data):
        posture = (
            PostureType[data.get("posture")]
            if data.get("posture") is not None
            else None
        )
        activity = (
            ActivityType[data.get("activity")]
            if data.get("activity") is not None
            else None
        )
        return Activity(activity=activity, posture=posture)
