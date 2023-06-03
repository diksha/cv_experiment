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

import inspect
import typing

from loguru import logger

from core.common.functional.lib.transform import Transform


def validate_transforms(transform_list: typing.List[Transform]):
    """
    Validates the transforms and inspects the type signatures of each

    Args:
        transform_list (typing.List[Transform]): the transform list to be composed

    Raises:
        ValueError: if the item in the list is not a transform
        ValueError: if there are more than one input to a nested set of transforms
        ValueError: if the function signatures do not match for the nested transforms
    """
    for transform in transform_list:
        if not isinstance(transform, Transform):
            raise ValueError(f"{transform.__name__} is not a transform")

    # ensure the output of one transform can be the input to the next
    if len(transform_list) > 1:
        for last_transform, current_transform in zip(
            transform_list[:-1], transform_list[1:]
        ):
            current_args = [
                arg
                for arg in inspect.getfullargspec(
                    current_transform.__call__
                ).args
                if arg != "self"
            ]
            if len(current_args) != 1:
                logger.error(f"Found issue with {current_transform}")
                raise ValueError(
                    "The nested transform cannot have more than one input argument!"
                )
            input_argument = current_args[0]

            last_return_type = typing.get_type_hints(
                last_transform.__call__
            ).get("return")
            current_input_type = typing.get_type_hints(
                current_transform.__call__
            ).get(input_argument)
            if (
                last_return_type is typing.Any
                or current_input_type is typing.Any
            ):
                continue
            if last_return_type is not current_input_type:
                logger.info(
                    " ".join(
                        (
                            "Found invalid signature for composition current type:",
                            f"{last_return_type} followed by: {current_input_type}",
                        )
                    )
                )
                raise ValueError(
                    " ".join(
                        (
                            f"The Transform's signature for {type(last_transform)}",
                            f"did not match the next transform: {type(current_transform)}",
                        )
                    )
                )
