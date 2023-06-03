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

# TODO(harishma): This class needs more work.
class Vector:
    def generate_vectors_from_points(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        v1_x = x2 - x1
        v1_y = y2 - y1
        v2_x = x4 - x3
        v2_y = y4 - y3
        v1 = [v1_x, v1_y]
        v2 = [v2_x, v2_y]
        return v1, v2
