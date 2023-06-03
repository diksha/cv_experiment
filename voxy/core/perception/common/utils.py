#
# Copyright 2020-2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import numpy as np
import torch

from core.structs.attributes import Polygon, RectangleXCYCWH


def reshape_polygon_crop_to_square(
    polygon: Polygon, frame: np.array
) -> torch.Tensor:
    """Reshapes a polygon crop to be a square

    Args:
        polygon (Polygon): polygon to convert to square
        frame (np.array): Image to process

    Returns:
        Image: cropped actor image in a square shape
    """

    # Converting the data to the same format as Bilal's inference
    # Predicted boxes are of the form List[(x_1, y_1, x_2, y_2)]
    inp_img = frame
    rect = RectangleXCYCWH.from_polygon(polygon)

    # Make square and pad by 10px on each side
    side = max(rect.w, rect.h) + 20
    rect.w = side
    rect.h = side

    # Adjust the center vertex to ensure polygon does not go out
    # of bounds
    rect_poly = rect.to_polygon()
    rect.center_vertice.x += min(
        0, inp_img.shape[1] - rect_poly.get_bottom_right().x
    ) - min(0, rect_poly.get_top_left().x)
    rect.center_vertice.y += min(
        0, inp_img.shape[0] - rect_poly.get_bottom_right().y
    ) - min(0, rect_poly.get_top_left().y)

    # Reconvert to polygon and return 4 params
    rect_poly = rect.to_polygon()

    return (
        int(rect_poly.get_top_left().y),
        int(rect_poly.get_bottom_right().y),
        int(rect_poly.get_top_left().x),
        int(rect_poly.get_bottom_right().x),
    )
