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


class _Item:
    pass


# poor man's recursive namespace
def set_recursive_simple_namespace(
    obj: typing.Type[object], items: dict
) -> None:
    """
    Mutates an object in place to produce a nested recursive namespace

    Args:
        obj (typing.Type[object]): object to mutate
        items (dict): the items to import into the object
    """

    def __convert_item_to_namespace(item: typing.Any) -> _Item:
        """
        Converts an keyword dictionary to a recursive namespace object

        Args:
            item (typing.Any): any item that needs to be added

        Returns:
            _Item: the converted namespace item
        """
        if not isinstance(item, dict) and not isinstance(item, list):
            return item
        subobj = _Item()
        set_recursive_simple_namespace(subobj, item)
        return subobj

    for key, value in items.items():
        if isinstance(value, dict):
            setattr(obj, key, __convert_item_to_namespace(value))
        elif isinstance(value, list):
            setattr(
                obj, key, [__convert_item_to_namespace(item) for item in value]
            )
        else:
            setattr(obj, key, value)


class RecursiveSimpleNamespace:
    """
    Simple Recursive namespace to recursively add all members of the class as
    attributes of the parent class. So if you have a input set of kwargs like:
    {"foo": 1, "bar": {"a": 2, "b":3}}

    You should be able to index the resulting object as:

    object.foo # 1
    object.bar.a # 2
    object.bar.b # 3
    """

    def __init__(self, **kwargs):
        self.__items = kwargs
        set_recursive_simple_namespace(self, kwargs)

    def to_dict(self) -> typing.Dict:
        """
        Grabs the items that were originally passed into construct
        the namespace

        Returns:
            typing.Dict: the kwargs dictionary
                         used when constructing the object
        """
        return self.__items
