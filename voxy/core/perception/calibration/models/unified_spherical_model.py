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
from core.structs.intrinsics import Intrinsics


def undistort(ux: float, uy: float, intrinsic_distortion: Intrinsics) -> tuple:
    """undistort.

        undistorts image points given an intrinsic calibration
        implementation is based on these papers:

        https://eg.uc.pt/bitstream/10316/4060/1/file159cef66b1ca4c17974f19f367f0f373.pdf

        The equations show up on page 4-5 of this paper as well

        https://tinyurl.com/mujs7xm4

    Args:
        ux (float): ux the image point in the x direction
        uy (float): uy the image point in the y direction
        intrinsic_distortion (Intrinsics): intrinsic_distortion the intrinsics stuct

    Returns:
        tuple[float, float]:
    """
    # TODO: actually add the model in here
    return (ux, uy)
