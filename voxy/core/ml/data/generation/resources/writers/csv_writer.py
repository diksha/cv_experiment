import os
import uuid
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from loguru import logger

from core.ml.data.generation.common.registry import WriterRegistry
from core.ml.data.generation.common.writer import (
    Collector,
    LabeledDataCollection,
    Writer,
    WriterMetaData,
)
from core.ml.data.generation.resources.streams.synchronized_readers import (
    DataCollectionMetaData,
)
from core.structs.dataset import DatasetFormat
from core.utils.aws_utils import upload_fileobj_to_s3


class CSVCollector(Collector):
    def __init__(
        self,
        output_directory: str,
        output_cloud_bucket: str,
        output_cloud_prefix: str,
    ):
        self.output_directory = output_directory
        self.output_cloud_bucket = output_cloud_bucket
        self.output_cloud_prefix = output_cloud_prefix
        self.data = []
        Path(output_directory).mkdir(exist_ok=True, parents=True)

    @staticmethod
    def _get_name(metadata: DataCollectionMetaData):
        name = "_".join(
            [
                metadata.source_name.replace("/", "_"),
                str(metadata.index),
                str(uuid.uuid4()),
            ]
        ).replace("-", "_")
        return name

    def collect_and_upload_data(
        self,
        data: np.ndarray,
        label: str,
        is_test: bool,
        metadata: DataCollectionMetaData,
    ):
        """
        Collects data and label for aggregation
        Args:
            data (np.ndarray): image data
            label (str): label for image
            is_test (bool): test/train flag
            metadata (DataCollectionMetaData): metadata associated to the image data
        """
        image_name = f"{self._get_name(metadata)}.png"
        image_bytes = cv2.imencode(".png", data)[1]
        s3_path = f"s3://{self.output_cloud_bucket}/{self.output_cloud_prefix}/{image_name}"
        was_written = upload_fileobj_to_s3(s3_path, image_bytes, "image/png")
        if was_written:
            self.data.append(
                (image_name, label, "test" if is_test else "train")
            )
        else:
            logger.warning("Image was not able to be written!")

    def dump(self) -> LabeledDataCollection:
        """
        Dump all collected data
        Returns:
            LabeledDataCollection: data used for writing aggregate dataset
        """
        return LabeledDataCollection(
            local_directory=self.output_directory,
            output_cloud_bucket=self.output_cloud_bucket,
            output_cloud_prefix=self.output_cloud_prefix,
            labeled_data=self.data,
        )


@WriterRegistry.register()
class CSVWriter(Writer):
    def __init__(
        self,
        output_directory: str,
        output_cloud_bucket: str = "voxel-datasets",
        output_cloud_prefix: str = "",
    ):
        self.output_directory = output_directory
        self.output_cloud_bucket = output_cloud_bucket
        self.output_cloud_prefix = os.path.join(
            output_cloud_prefix, str(uuid.uuid4())
        )

    def create_collector(self) -> CSVCollector:
        """Creates an instance of CSVCollector
        Returns:
            CSVCollector: csv collector for generating dataset from items
        """
        return CSVCollector(
            output_directory=self.output_directory,
            output_cloud_bucket=self.output_cloud_bucket,
            output_cloud_prefix=self.output_cloud_prefix,
        )

    def _flatten_labeled_data_collections(
        self, labeled_data_collections: List[LabeledDataCollection]
    ) -> List[Tuple[str, str, str]]:
        """Flattens the collected list of LabeledDataCollections
        Args:
            labeled_data_collections (List[LabeledDataCollection]): list of aggregated
                label collections
        Returns:
            List[Tuple[str, str, str]]: list of labels to write in csv format
        Raises:
            RuntimeError: if LabeledDataCollection were written to different locations
        """
        labeled_data = []
        for labeled_data_collection in labeled_data_collections:
            labeled_data.extend(labeled_data_collection.labeled_data)

        return labeled_data

    def write_dataset(
        self, labeled_data_collections: List[LabeledDataCollection]
    ) -> WriterMetaData:
        """
        Write csv data set to common cloud path
        Args:
            labeled_data_collections (List[LabeledDataCollection]): aggregated labeled
                data collections to convert into a dataset
        Returns:
            WriterMetaData: CSV Writer meta data containing local directory to download dataset
                and the cloud path where the data is located
        """
        labeled_data = self._flatten_labeled_data_collections(
            labeled_data_collections
        )
        dataset = pd.DataFrame(
            labeled_data, columns=["image", "label", "is_test"]
        )
        logger.info(f"Number of unique labels {dataset.shape[0]}")
        csv_name = "labels.csv"

        # upload to S3
        s3_path = f"s3://{self.output_cloud_bucket}/{self.output_cloud_prefix}/{csv_name}"
        upload_fileobj_to_s3(
            s3_path,
            dataset.to_csv(
                mode="w",
                header=False,
                index=False,
            ).encode("utf-8"),
            "text/csv",
        )
        return WriterMetaData(
            local_directory=self.output_directory,
            cloud_path=(
                f"s3://{self.output_cloud_bucket}/"
                f"{self.output_cloud_prefix}"
            ),
        )

    @classmethod
    def format(cls) -> DatasetFormat:
        """
        Returns the format type of the dataset
        for dataloader compatibility

        Returns:
            DatasetFormat: the format of the dataset
        """
        return DatasetFormat.IMAGE_CSV
