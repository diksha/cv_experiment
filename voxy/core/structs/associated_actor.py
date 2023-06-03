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
import attr

from .actor import Actor


@attr.s(slots=True)
class AssociatedActor:

    associated_frame = attr.ib(type="AssociatedFrame")
    iou = attr.ib(type=float)
    gt = attr.ib(type=Actor)
    pred = attr.ib(type=Actor)
    derived = attr.ib(type=dict, factory=dict)

    def set_derived(self, key, value):
        assert key not in self.derived, f"{key} already exists in derived"
        self.derived[key] = value
