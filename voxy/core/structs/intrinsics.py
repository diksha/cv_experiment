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


# trunk-ignore-all(mypy)
import attr
import numpy as np


@attr.s(slots=True)
class Intrinsics:

    # the focal length is defined in pixels for an image of size 299x299
    # for more information on focal lengths:
    #
    # https://www.cs.ccu.edu.tw/~damon/tmp/SzeliskiBook_20100903_draft.pdf#page=75
    focal_length_px: float = attr.ib()

    # the distortion parameter is defined in the unified spherical model
    # reference: https://eg.uc.pt/bitstream/10316/4060/1/file159cef66b1ca4c17974f19f367f0f373.pdf
    # A unifying geometric representation for central projection systems
    # JP Barreto - Computer Vision and Image Understanding, 2006
    #
    # reference: https://eg.uc.pt/bitstream/10316/4060/1/file159cef66b1ca4c17974f19f367f0f373.pdf
    distortion_parameter_usm: float = attr.ib()

    cx: float = attr.ib()
    cy: float = attr.ib()

    width_px: float = attr.ib()
    height_px: float = attr.ib()

    def to_calibration_matrix(self) -> np.array:
        """Generate and return calibration matrix for intrinsic parameters.

        Args:

        Returns:
            np.array: 4x4 calibration matrix based on the given intrinsic parameters
        """
        # The focal length is expressed in pixel coordinates from deepcalib for a 299x299 image
        # the conversion is applied to convert it first into unitless coordinates, then into those
        # expressed in pixels under the resolution of the camera
        #
        # reference: https://www.cs.ccu.edu.tw/~damon/tmp/SzeliskiBook_20100903_draft.pdf#page=75
        fx = fy = self.focal_length_px / 299 * self.width_px / 2
        intrinsic_matrix = np.array(
            [
                [fx, 0.0, self.cx, 0.0],
                [0.0, fy, self.cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return intrinsic_matrix
