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

from enum import Enum, unique

from core.common.utils.recursive_namespace import RecursiveSimpleNamespace


@unique
class TaskPurpose(Enum):
    """
    Task purpose defines what functional need a model is
    designed to address for the perception system

    """

    UNKNOWN = 0
    OBJECT_DETECTION_2D = 1
    PPE_SAFETY_VEST = 2
    PPE_HARD_HAT = 3
    DOOR_STATE = 4
    HUMAN_KEYPOINT_DETECTION_2D = 5
    SPILL = 6
    ERGONOMIC_BAD_LIFT = 7
    ERGONOMIC_OVERREACH = 8
    PPE_SAFETY_GLOVES = 9
    ERGONOMIC_CARRY_OBJECT = 10
    PPE_BUMP_CAP = 11

    @staticmethod
    def names() -> list:
        """
        Get the valid list of task names.

        Returns:
            list: the valid list of task name strings
        """
        # exclude unknown category
        return [
            member.name
            for member in TaskPurpose
            if member != TaskPurpose.UNKNOWN
        ]


class Task(RecursiveSimpleNamespace):
    """
    High level wrapper for task. Reflection of instance in
    GraphQL Metaverse
    """
