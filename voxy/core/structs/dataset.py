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

import os
import typing
from enum import Enum, unique

from loguru import logger

from core.common.utils.recursive_namespace import RecursiveSimpleNamespace
from core.utils.aws_utils import (
    download_directory_from_s3,
    get_bucket_path_from_s3_uri,
)

# These structs use the RecursiveSimpleNamespace so they have auto-populated fields
# Must suppress all `Instance of <> has no <> member`
# trunk-ignore-all(pylint/E1101)


class DatasetFormat(Enum):
    """
    Dataset formats define what format and writer was used
    when generating the dataset. This is used downstream to
    configure and load datasets in different formats
    """

    UNKNOWN = 0
    IMAGE_CSV = 1
    YOLO_V5_CONFIG = 2  # image text files and yolov5 yaml config

    @staticmethod
    def names() -> typing.List[str]:
        """
        Generates the list of valid dataset formats

        Returns:
            typing.List[str]: the valid list of dataset format
        """
        return [
            member.name
            for member in DatasetFormat
            if member != DatasetFormat.UNKNOWN
        ]


@unique
class DataloaderType(Enum):
    """
    Dataloader type defines the dataloader required to train the model
    """

    UNKNOWN = 0
    PYTORCH_IMAGE = 1
    NUMPY_IMAGE = 2


class DataCollectionLogset(RecursiveSimpleNamespace):
    """
    Simple wrapper class for data collection logset and all it's helper
    methods
    """


class Dataset(RecursiveSimpleNamespace):
    """
    Simple Dataset class wrapper used when grabbing from metaverse
    in the graphql response
    """

    def __init__(self, local_path=None, downloaded=False, **kwargs):
        super().__init__(**kwargs)
        # some nested logic

        # TODO: remove this nested logic and add some extra logic for just the data collection
        self.logset = self._get_logset()
        self.downloaded = downloaded and local_path
        self.local_path = local_path

    def _get_logset(self) -> DataCollectionLogset:
        """
        Grabs the logset from the dataset response

        Returns:
            DataCollectionLogset: the data collection logset for the dataset
        """
        return self.data_collection_logset_ref

    def set_local_download_path(self, local_path: str):
        """
        Sets the local download path

        Args:
            local_path (str): the path where the dataset is downloaded
        """
        self.downloaded = True
        self.local_path = local_path

    def get_download_path(self) -> typing.Optional[str]:
        """
        Grabs the downloaded path if it is present, otherwise
        returns None

        Returns:
            typing.Optional[str]: _description_
        """
        return self.local_path if self.downloaded else None

    def is_downloaded(self) -> bool:
        """
        Simple getter to see if a dataset object has been downloaded

        Returns:
            bool: whether it is downloaded
        """
        return self.downloaded

    def download(self, local_path: str = "/data") -> str:
        """
        Downloads the dataset to the local path provided

        Args:
            local_path (str): the local path of the dataset

        Returns:
            str: the download location
        """
        if (
            self.downloaded
            and local_path == self.local_path
            and os.path.exists(local_path)
            and len(os.listdir(local_path)) > 1
        ):
            logger.info("Dataset already present")
            return self.local_path
        logger.info(f"Downloading dataset to {local_path} {self.path}")
        bucket, prefix = get_bucket_path_from_s3_uri(self.path)
        download_directory_from_s3(local_path, bucket, prefix)
        return local_path
