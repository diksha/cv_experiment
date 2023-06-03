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

from core.common.utils.recursive_namespace import RecursiveSimpleNamespace
from core.structs.frame import Frame


@unique
class DataCollectionType(Enum):
    """Type of data collection

    Args:
        Enum : parent type
    """

    UNKNOWN = 0
    VIDEO = 1
    IMAGE_COLLECTION = 2

    @staticmethod
    def names() -> list:
        """
        Returns all valid data collection type enum names

        Returns:
            list: the list of data collection types as a string list
        """
        # exclude unknown category
        return [
            member.name
            for member in DataCollectionType
            if member != DataCollectionType.UNKNOWN
        ]


class DataCollection(RecursiveSimpleNamespace):
    """
    Simple wrapper class for data collection and all it's helper
    methods.

    To be used only for offline processes.
    DO NOT USE in develop or production graph
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "frame_ref" in kwargs:
            self.frames = [
                Frame.from_metaverse(frame) for frame in kwargs["frame_ref"]
            ]

    @classmethod
    def from_metaverse(cls, contents: dict) -> "DataCollection":
        """
        Unpacks the contents dictionary and generates a new data collection object

        Args:
            contents (dict): the dictionary of information to be loaded into the data collection

        Returns:
            DataCollection: the new data collection generated from the dictionary
        """
        return DataCollection(**contents)
