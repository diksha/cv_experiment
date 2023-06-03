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
from typing import List


def get_polygon_vertices(frame) -> List:
    """Given scale bounding boxes, get polygon

    Args:
        frame (dict): bounding box frame

    Returns:
        List: vertices
    """
    return [
        {"x": frame["left"], "y": frame["top"], "z": None},
        {
            "x": frame["left"] + frame["width"],
            "y": frame["top"],
            "z": None,
        },
        {
            "x": frame["left"] + frame["width"],
            "y": frame["top"] + frame["height"],
            "z": None,
        },
        {
            "x": frame["left"],
            "y": frame["top"] + frame["height"],
            "z": None,
        },
    ]
