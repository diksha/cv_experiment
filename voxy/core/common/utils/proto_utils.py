#
# Copyright 2022 Voxel Labs, Inc.
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

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message as ProtoMessage
from loguru import logger


def is_repeated_field(descriptor_table: dict, attribute: str) -> bool:
    """
    Checks if the item is a repeated field. You cannot check
    for the existence of a field if it is repeated

    Args:
        descriptor_table (dict): the descriptor table describing the attributes of the proto
        attribute (str): the attribute of the

    Returns:
        bool: whether the field is repeated or not
    """
    return descriptor_table.get(attribute) == FieldDescriptor.LABEL_REPEATED


def is_required_field(descriptor_table: dict, attribute: str) -> bool:
    """
    Checks if the item is a required field. You cannot check
    for the existence of a field if it is required

    Args:
        descriptor_table (dict): the descriptor table describing the attributes of the proto
        attribute (str): the attribute of the

    Returns:
        bool: whether the field is required or not
    """
    return descriptor_table.get(attribute) == FieldDescriptor.LABEL_REQUIRED


class VoxelProto:
    """
    Wraps Protobufs for easier member access

    accessing members triggers the default
    constructor of proto3 protobufs so this
    returns optional values so it works pythonically
    and does not update optional values in the proto
    upon access
    """

    def __init__(self, proto: ProtoMessage):
        """
        Initializes the proto wrapper

        Args:
            proto (ProtoMessage): protobuf to wrap
        """
        self.data = proto

        self.descriptor_table = {
            descriptor.name: descriptor.label
            for descriptor in self.data.DESCRIPTOR.fields
        }

    def __repr__(self) -> str:
        """
        Shows a string representation of the
        protobuf

        Returns:
            str: the string representation
        """
        return repr(self.data)

    def __str__(self) -> str:
        """
        Generates string from protobuf

        Returns:
            str: the string version of the proto
        """
        return str(self.data)

    def __getattr__(self, attribute: str) -> typing.Any:
        """
        Gets the value of the protobuf because
        accessing members triggers the default
        constructor of proto3 protobufs

        Args:
            attribute (str): the name of the attribute to grab

        Returns:
            typing.Any: the value from the attribute or None if it
                        does not exist
        """
        if is_repeated_field(self.descriptor_table, attribute):
            value = getattr(self.data, attribute)
            return value

        if is_required_field(self.descriptor_table, attribute):
            value = getattr(self.data, attribute)
            return value

        try:
            if self.data.HasField(attribute):
                value = getattr(self.data, attribute)
                return value
            return None
        except ValueError:
            # this happens when the optional field is at the
            # top level of the proto.. there is no documentation
            # on how to circumvent this issue unfortunately
            logger.warning(f"safe checking attribute: {attribute} failed")
            return getattr(self.data, attribute)

    def to_proto(self) -> ProtoMessage:
        """
        Returns the underlying protobuf
        datastructure

        Returns:
            ProtoMessage: the protomessage
        """
        return self.data
