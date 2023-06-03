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

from enum import Enum, unique

# proto imports fail with trunk
# trunk-ignore-begin(pylint/E0611)
from protos.perception.structs.v1.model_pb2 import (
    ModelConfiguration as ModelConfigurationPb,
)

# trunk-ignore-end(pylint/E0611)


class ModelConfiguration:
    """
    Model configuration required for generating models
    and configuring transforms

    This is a helper class to attach methods to the protobuf data
    we have in the protobuf message such as serialization/deserialization
    and conversion methods
    """

    def __init__(self, proto_message: ModelConfigurationPb):
        self.data = proto_message

    @classmethod
    def from_proto(
        cls, proto_message: ModelConfigurationPb
    ) -> "ModelConfiguration":
        """
        Converts the model configuration to a protobuf

        Args:
            proto_message (ModelConfigurationPb): the protobuf message

        Returns:
            ModelConfiguration: the model configuration generated
                      from the protobuf
        """
        return cls(
            proto_message=proto_message,
        )

    def to_proto(self) -> ModelConfigurationPb:
        """
        Converts the model configuration to a protobuf

        Returns:
            ModelConfigurationPb: the generated protobuf
        """
        return self.data


@unique
class ModelCategory(Enum):
    UNKNOWN = 0
    IMAGE_CLASSIFICATION = 1
    IMAGE_SEGMENTATION = 2
    SEGMENTATION = 3
    OBJECT_DETECTION = 4

    @staticmethod
    def names():
        # exclude unknown category
        return [member.name for member in ModelCategory if member.value > 0]


# TODO(twroge): define service version in a way that prevents users from making an error
#
#              Requirements:
#                1. Update the service version based on if the architecture has changed
#                2. Update the service version based on if the inputs have changed (video/image)
#                3. Update the version based on if the parameters change for the graph config
