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


class RegistrationException(Exception):
    """
    Exception raised in registration or grabbing from the BaseClassRegistry
    """


class BaseClassRegistry:
    """
    Basic class registry decorator and wrapper.
    """

    @classmethod
    def get_registry(cls):
        """
        Returns the registry for the child class. This should be a static member

        Raises:
            NotImplementedError: if the registry has not been implemented in the child class
        """
        raise NotImplementedError(
            "The baseclass cannot have a static registry. \
                This should be implemented in the child class"
        )

    @classmethod
    def register(cls) -> typing.Any:
        """
        Registers the class into the base registry with the key being the
        name of the item and the item being the class itself

        Returns:
            typing.Any: the decorator that registers the class
        """

        def wrapper(item: typing.Any) -> typing.Any:
            """
            Registration decorator. This is designed to take in and return a class object

            Args:
                item (typing.Any): the base class to register

            Returns:
                typing.Any: the wrapped class
            """
            cls.get_registry()[item.__name__] = item
            return item

        return wrapper

    @classmethod
    def get_instance(cls, name: str, params: dict) -> typing.Any:
        """
        Grabs the class from the registry and returns the constructed instance
        with arguments passed in

        Args:
            name (str): the name of the registered item
            params (dict): the arguments to pass to the registered item

        Raises:
            RegistrationException: if the item is not found in the registry

        Returns:
            typing.Any: the constructed instance from the registry
        """
        item = cls.get_registry().get(name)
        if item is None:
            raise RegistrationException(
                f"{name} not found in available registered keys: {cls.get_registry().keys()}"
            )

        return item(**params)

    @classmethod
    def get_class(cls, name: str) -> typing.Any:
        """
        Grabs the class from the registry and returns it

        Args:
            name (str): the name of the registered item

        Raises:
            RegistrationException: if the item is not found in the registry

        Returns:
            typing.Any: the class with name, `name`
        """
        item = cls.get_registry().get(name)
        if item is None:
            raise RegistrationException(
                f"{name} not found in available registered keys: {cls.get_registry().keys()}"
            )

        return item
