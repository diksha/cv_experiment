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

import toolz

from core.common.functional.lib.transform import Transform
from core.common.functional.lib.utils import validate_transforms


# trunk-ignore(pylint/C0103)
def Compose(transform_list: typing.List[Transform]) -> typing.Callable:
    """
    Composes a list of transforms

    Args:
        transform_list (list): the list of transforms to compose. functions get composed as:
                            g(f(x)) for args [f, g] so the output of f goes into the input of g

    Returns:
        typing.Callable: the callable function created by composing the sub functions
    """
    validate_transforms(transform_list=transform_list)
    return toolz.compose(*list(reversed(transform_list)))
