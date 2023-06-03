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

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List

from core.ml.data.generation.common.metadata import MetaData
from core.structs.dataset import DatasetFormat


@dataclass
class LabeledDataCollection(MetaData):
    """
    Any specific labeled data collector metadata associated
    with the the buffered collection operations

    Used by the writer to upload the aggregated dataset
    """

    local_directory: str
    output_cloud_bucket: str
    output_cloud_prefix: str
    labeled_data: List[object]

    def dump(self) -> dict:
        """
        Dumps specific data associated with this instance
        of labeld data collection metadata
        (required for parent MetaData)

        Returns:
            dict: the dict of all attributes of this instance
        """
        return asdict(self)


@dataclass
class WriterMetaData(MetaData):
    """
    Any specific writer metadata associated with
    the current buffered writing operations

    Used when uploading to S3 after the writer is closed
    """

    local_directory: str
    cloud_path: str

    def dump(self) -> dict:
        """
        Dumps specific data associated with this instance
        of writer metadata (required for parent MetaData)

        Returns:
            dict: the dict of all attributes of this instance
        """
        return asdict(self)


class Collector(ABC):
    """
    Collector is responsible for collecting data and labels from a single source
    into a format (list) that can be aggregated from multiple jobs into a complete
    dataset that can be used for training
    """

    @abstractmethod
    def collect_and_upload_data(self, data, label, is_test, metadata):
        """
        Collects data and its corresponding label, split, and metadata for aggregation.
        Uploads required data to the cloud.

        Args:
            data (np.ndarray): data to be collected
            label (object): label associated to the data
            is_test (bool): train/test flag
            metadata (object): metadata associated to the data
        Raises:
            NotImplementedError: when base class method has not been implemented
        """
        raise NotImplementedError("Collector must implement collect method")

    def dump(self) -> LabeledDataCollection:
        """
        Dumps the labeled data collected over the course of the collectors existance
        so that it can be written as a dataset.
        """
        raise NotImplementedError("Collector must implement dump method")


class Writer(ABC):
    """
    Writer is responsible for unifying the data and labels into some serializable format
    (like coco/csv etc.) that can be used for training
    """

    @abstractmethod
    def create_collector(self) -> Collector:
        """Instantiates a Collector corresponding to the writer
        Raises:
            NotImplementedError: when base class method has not been implemented
        """
        raise NotImplementedError(
            "Writer must implement create_collector method"
        )

    @abstractmethod
    def write_dataset(
        self, labeled_data_collections: List[LabeledDataCollection]
    ) -> WriterMetaData:
        """
        Takes aggregated labeled data collections and converts into a dataset usable for model
        training

        Args:
            labeled_data_collections (List[LabeledDataCollection]): aggregated labeled
                data collections
        Raises:
            NotImplementedError: when base class method has not been implemented
        """
        raise NotImplementedError("Writer must implement write_dataset method")

    @classmethod
    @abstractmethod
    def format(cls) -> DatasetFormat:
        """
        Returns the format type of the dataset
        for dataloader compatibility

        Returns:
            DatasetFormat: the format of the dataset
        """


# trunk-ignore(flake8/W292)
