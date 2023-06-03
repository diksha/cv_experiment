#
# Copyright 2023 Voxel Labs, Inc.
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
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

from core.infra.sematic.perception.yolo.yolo_options import (
    CONFIG_DIR_NAME,
    IMAGE_DIR_NAME,
    IMAGE_EXT,
    LABEL_DIR_NAME,
    LABEL_EXT,
    TEST_DIR_NAME,
    TRAIN_DIR_NAME,
    DataSplit,
    get_yolo_config_name,
    get_yolo_dataset_name,
)
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
from core.structs.actor import get_ordered_actors
from core.structs.dataset import DatasetFormat
from core.utils.aws_utils import upload_fileobj_to_s3


class YOLOCollector(Collector):
    def __init__(
        self,
        actor_categories: List[str],
        output_directory: str,
        output_cloud_bucket: str,
        output_cloud_prefix: str,
    ):
        self.output_directory = output_directory
        self.output_cloud_bucket = output_cloud_bucket
        self.output_cloud_prefix = output_cloud_prefix
        Path(output_directory).mkdir(exist_ok=True, parents=True)
        for evaluation_type in [TRAIN_DIR_NAME, TEST_DIR_NAME]:
            for data_type in [IMAGE_DIR_NAME, LABEL_DIR_NAME, CONFIG_DIR_NAME]:
                directory_to_create = (
                    f"{output_directory}/{evaluation_type}/{data_type}"
                )
                Path(directory_to_create).mkdir(exist_ok=True, parents=True)
        self.actor_categories = [
            category.name for category in get_ordered_actors(actor_categories)
        ]
        self.train_labels = {}
        self.test_labels = {}

    @staticmethod
    def get_name(metadata: DataCollectionMetaData) -> str:
        """Gets the name of the image using the metadata

        Args:
            metadata (DataCollectionMetaData): metadata associated to the image

        Returns:
            str: name of the image, which will be a flattened video uuid, along with frame_tms
                appended as the suffix
        """
        name = "_".join(
            [
                metadata.source_name.replace("/", "_"),
                "frame",
                str(metadata.time_ms),
            ]
        ).replace("-", "_")
        return name

    def _write_annotations(
        self, labeled_data: Dict[str, List[str]], split_dir_name: str
    ):
        """Write yolo annotations to the text file

        Args:
            labeled_data (Dict[str, List[str]]): dictionary where the key is the image name
                (which is used as part of the file name), and the value is a list of object
                annotations that need to be written into a text file
            split_dir_name (str): train / test split directory, used to save annotations to
                their respective folder
        """
        for data_name, annotation_list in labeled_data.items():
            annotation_name = f"{data_name}.{LABEL_EXT}"
            annotations = "\n".join(annotation_list)
            s3_path = (
                f"s3://{self.output_cloud_bucket}/"
                f"{self.output_cloud_prefix}/{split_dir_name}/"
                f"{LABEL_DIR_NAME}/{annotation_name}"
            )
            upload_fileobj_to_s3(
                s3_path, annotations.encode("utf-8"), "text/plain"
            )

    def _write_image(
        self, data: np.ndarray, name: str, split_dir_name: str
    ) -> bool:
        """Write image data to s3

        Args:
            data (np.ndarray): image data to be written
            name (str): name of image being written
                (not full path, with extension)
            split_dir_name (str): test/train split dir name

        Returns:
            bool: successful upload to s3
        """
        image_bytes = cv2.imencode(".png", data)[1]
        s3_path = (
            f"s3://{self.output_cloud_bucket}/"
            f"{self.output_cloud_prefix}/{split_dir_name}/"
            f"{IMAGE_DIR_NAME}/{name}"
        )
        return upload_fileobj_to_s3(
            s3_path,
            image_bytes,
            "image/png",
        )

    def collect_and_upload_data(
        self,
        data: np.ndarray,
        label: str,
        is_test: bool,
        metadata: DataCollectionMetaData,
    ):
        # Figure out which split data goes in
        split = TRAIN_DIR_NAME
        labeled_data = self.train_labels
        if is_test:
            split = TEST_DIR_NAME
            labeled_data = self.test_labels
        # Write data if its first time we see it
        data_name = self.get_name(metadata)
        image_name = f"{data_name}.{IMAGE_EXT}"
        if data_name not in labeled_data:
            was_written = self._write_image(
                data=data, name=image_name, split_dir_name=split
            )
            if was_written:
                labeled_data[data_name] = []
            else:
                logger.warning("Image was not able to be written!")
                labeled_data[data_name] = None
        if labeled_data[data_name] is not None:
            labeled_data[data_name].append(label)

    def dump(self) -> LabeledDataCollection:
        """
        Dump all collected data
        Returns:
            LabeledDataCollection: data used for writing aggregate dataset
        """
        self._write_annotations(
            labeled_data=self.train_labels, split_dir_name=TRAIN_DIR_NAME
        )
        self._write_annotations(
            labeled_data=self.test_labels, split_dir_name=TEST_DIR_NAME
        )
        return LabeledDataCollection(
            local_directory=self.output_directory,
            output_cloud_bucket=self.output_cloud_bucket,
            output_cloud_prefix=self.output_cloud_prefix,
            labeled_data={
                TRAIN_DIR_NAME: list(self.train_labels.keys()),
                TEST_DIR_NAME: list(self.test_labels.keys()),
            },
        )


@WriterRegistry.register()
class YOLOWriter(Writer):
    def __init__(
        self,
        actor_categories: List[str],
        output_directory: str,
        output_cloud_bucket: str = "voxel-datasets",
        output_cloud_prefix: str = "derived/voxel/yolo",
    ):
        self.output_directory = output_directory
        self.output_cloud_bucket = output_cloud_bucket
        self.output_cloud_prefix = os.path.join(
            output_cloud_prefix, str(uuid.uuid4())
        )
        Path(output_directory).mkdir(exist_ok=True, parents=True)
        for evaluation_type in [TRAIN_DIR_NAME, TEST_DIR_NAME]:
            for data_type in [IMAGE_DIR_NAME, LABEL_DIR_NAME, CONFIG_DIR_NAME]:
                directory_to_create = (
                    f"{output_directory}/{evaluation_type}/{data_type}"
                )
                Path(directory_to_create).mkdir(exist_ok=True, parents=True)
        self.actor_categories = [
            category.name for category in get_ordered_actors(actor_categories)
        ]

    def create_collector(self) -> YOLOCollector:
        """Creates an instance of YOLOCollector
        Returns:
            YOLOCollector: csv collector for generating dataset from items
        """
        return YOLOCollector(
            actor_categories=self.actor_categories,
            output_directory=self.output_directory,
            output_cloud_bucket=self.output_cloud_bucket,
            output_cloud_prefix=self.output_cloud_prefix,
        )

    def _flatten_labeled_data_collections(
        self, labeled_data_collections: List[LabeledDataCollection]
    ) -> Tuple[List[str], List[str]]:
        """Flattens the collected list of LabeledDataCollections
        Args:
            labeled_data_collections (List[LabeledDataCollection]): list of aggregated
                label collections
        Returns:
            List[Tuple[str, str, str]]: list of labels to write in csv format
        Raises:
            RuntimeError: if LabeledDataCollection were written to different locations
        """
        train_data = []
        test_data = []
        for labeled_data_collection in labeled_data_collections:
            train_data.extend(
                labeled_data_collection.labeled_data[TRAIN_DIR_NAME]
            )
            test_data.extend(
                labeled_data_collection.labeled_data[TEST_DIR_NAME]
            )

        return train_data, test_data

    def _get_train_val_split(
        self, train_image_names: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Split training data into train and validation sets (80-20 split)

        Args:
            train_image_names (List[str]): image names for training that we want
                to split into train and validation sets

        Returns:
            Tuple(List[str], List[str]): tuple of image names used for train and validation
                sets, respectively
        """
        site_set = {
            "_".join(image_name.split("_")[0:2])
            for image_name in train_image_names
        }
        train_split = []
        val_split = []
        for site in site_set:
            site_images = list(
                {
                    image_name
                    for image_name in train_image_names
                    if site in image_name
                }
            )
            if len(site_images) > 1:
                train_images, val_images = train_test_split(
                    site_images, test_size=0.2, random_state=42
                )
                train_split.extend(train_images)
                val_split.extend(val_images)
            else:
                logger.warning(
                    f"Site {site} low on training images, {site_images}"
                )
                train_split.extend(site_images)
        return train_split, val_split

    def _write_dataset_file(
        self,
        image_names: List[str],
        split_dir_name: str,
        file_name: str,
        cloud_path_prefix: str,
    ):
        """Writes dataset text file, which is a line delimited list of full paths referencing
        the images downloaded on the machine that will be used for training.

        Args:
            image_names (List[str]): list of image namees that will be used to train yolo
            split_dir_name (str): train / test dir name, used to point yolo to the folder where
                the images exist
            file_name (str): file name of the dataset text file
            cloud_path_prefix (str): cloud path prefix where the text file will be uploaded
        """
        image_paths = [
            f"{self.output_directory}/{split_dir_name}/{IMAGE_DIR_NAME}/{image_name}.{IMAGE_EXT}"
            for image_name in sorted(image_names)
        ]
        dataset_text = "\n".join(image_paths)
        s3_path = (
            f"s3://{self.output_cloud_bucket}/{cloud_path_prefix}/{file_name}"
        )
        upload_fileobj_to_s3(
            s3_path, dataset_text.encode("utf-8"), "text/plain"
        )

    def _write_yolo_config(
        self,
        train_dataset_file_name: Union[str, List],
        val_dataset_file_name: Union[str, List],
        test_dataset_file_name: Union[str, List],
        root_dir_name: str,
        file_name: str,
        cloud_path_prefix: str,
    ):
        """
        Generates dataset config used by yolo to train and test dataset

        Args:
            train_dataset_file_name (Union[str, List]): train dataset file name
                (empty list means no file)
            val_dataset_file_name (Union[str, List]): val dataset file name
                (empty list means no file)
            test_dataset_file_name (Union[str, List]): test dataset file name
                (empty list means no file)
            root_dir_name (str): root dir of where to save config
            file_name (str): file name of the config
            cloud_path_prefix (str): cloud path prefix where the text file will be uploaded
        """
        yolo_config = {
            "names": self.actor_categories,
            "nc": len(self.actor_categories),
            "path": root_dir_name,
            "train": train_dataset_file_name,
            "val": val_dataset_file_name,
            "test": test_dataset_file_name,
        }
        s3_path = (
            f"s3://{self.output_cloud_bucket}/{cloud_path_prefix}/{file_name}"
        )
        upload_fileobj_to_s3(
            s3_path, yaml.dump(yolo_config).encode("utf-8"), "text/yaml"
        )

    def write_dataset(
        self, labeled_data_collections: List[LabeledDataCollection]
    ) -> WriterMetaData:
        """Writes an image if it has not been saved locally, and stores object annotation to the
        image annotation map

        Args:
            labeled_data_collections (List[LabeledDataCollection]): aggregated labeled
                data collections to convert into a dataset
        Returns:
            WriterMetaData: YOLOWriter meta data containing local directory to download dataset
                and the cloud path where the data is located
        """
        # Get train, val, and test datasets
        train_val_data, test_data = self._flatten_labeled_data_collections(
            labeled_data_collections
        )
        train_data, val_data = self._get_train_val_split(train_val_data)

        # Write dataset files and config for training
        train_root_dir = (
            f"{self.output_directory}/{TRAIN_DIR_NAME}/{CONFIG_DIR_NAME}"
        )
        train_cloud_prefix = (
            f"{self.output_cloud_prefix}/{TRAIN_DIR_NAME}/{CONFIG_DIR_NAME}"
        )
        train_dataset_name = get_yolo_dataset_name(DataSplit.TRAINING)
        val_dataset_name = get_yolo_dataset_name(DataSplit.VALIDATION)
        train_config_name = get_yolo_config_name(DataSplit.TRAINING)
        self._write_dataset_file(
            image_names=train_data,
            split_dir_name=TRAIN_DIR_NAME,
            file_name=train_dataset_name,
            cloud_path_prefix=train_cloud_prefix,
        )
        self._write_dataset_file(
            image_names=val_data,
            split_dir_name=TRAIN_DIR_NAME,
            file_name=val_dataset_name,
            cloud_path_prefix=train_cloud_prefix,
        )
        self._write_yolo_config(
            train_dataset_file_name=train_dataset_name,
            val_dataset_file_name=val_dataset_name,
            test_dataset_file_name=[],
            root_dir_name=train_root_dir,
            file_name=train_config_name,
            cloud_path_prefix=train_cloud_prefix,
        )

        # Write dataset files and configs for testing purposes
        test_root_dir = (
            f"{self.output_directory}/{TEST_DIR_NAME}/{CONFIG_DIR_NAME}"
        )
        test_cloud_prefix = (
            f"{self.output_cloud_prefix}/{TEST_DIR_NAME}/{CONFIG_DIR_NAME}"
        )
        test_dataset_name = get_yolo_dataset_name(DataSplit.TESTING)
        test_config_name = get_yolo_config_name(DataSplit.TESTING)
        self._write_dataset_file(
            image_names=test_data,
            split_dir_name=TEST_DIR_NAME,
            file_name=test_dataset_name,
            cloud_path_prefix=test_cloud_prefix,
        )
        self._write_yolo_config(
            train_dataset_file_name=[],
            val_dataset_file_name=[],
            test_dataset_file_name=test_dataset_name,
            root_dir_name=test_root_dir,
            file_name=test_config_name,
            cloud_path_prefix=test_cloud_prefix,
        )

        # Get splits by sites
        train_sites = {
            "_".join(image_name.split("_")[0:2]) for image_name in train_data
        }
        val_sites = {
            "_".join(image_name.split("_")[0:2]) for image_name in val_data
        }
        test_sites = {
            "_".join(image_name.split("_")[0:2]) for image_name in test_data
        }

        # Write dataset text files split up by site
        for train_site in train_sites:
            file_name = get_yolo_dataset_name(
                DataSplit.TRAINING, dataset_prefix=f"{train_site}_"
            )
            config_name = get_yolo_config_name(
                DataSplit.TRAINING, dataset_prefix=f"{train_site}_"
            )
            train_data_subset = list(
                {
                    image_name
                    for image_name in train_data
                    if train_site in image_name
                }
            )
            self._write_dataset_file(
                image_names=train_data_subset,
                split_dir_name=TRAIN_DIR_NAME,
                file_name=file_name,
                cloud_path_prefix=test_cloud_prefix,
            )
            self._write_yolo_config(
                train_dataset_file_name=[],
                val_dataset_file_name=[],
                test_dataset_file_name=file_name,
                root_dir_name=test_root_dir,
                file_name=config_name,
                cloud_path_prefix=test_cloud_prefix,
            )

        for val_site in val_sites:
            file_name = get_yolo_dataset_name(
                DataSplit.VALIDATION, dataset_prefix=f"{val_site}_"
            )
            config_name = get_yolo_config_name(
                DataSplit.VALIDATION, dataset_prefix=f"{val_site}_"
            )
            val_data_subset = list(
                {
                    image_name
                    for image_name in val_data
                    if val_site in image_name
                }
            )
            self._write_dataset_file(
                image_names=val_data_subset,
                split_dir_name=TRAIN_DIR_NAME,
                file_name=file_name,
                cloud_path_prefix=test_cloud_prefix,
            )
            self._write_yolo_config(
                train_dataset_file_name=[],
                val_dataset_file_name=[],
                test_dataset_file_name=file_name,
                root_dir_name=test_root_dir,
                file_name=config_name,
                cloud_path_prefix=test_cloud_prefix,
            )

        for test_site in test_sites:
            file_name = get_yolo_dataset_name(
                DataSplit.TESTING, dataset_prefix=f"{test_site}_"
            )
            config_name = get_yolo_config_name(
                DataSplit.TESTING, dataset_prefix=f"{test_site}_"
            )
            test_data_subset = list(
                {
                    image_name
                    for image_name in test_data
                    if test_site in image_name
                }
            )
            self._write_dataset_file(
                image_names=test_data_subset,
                split_dir_name=TEST_DIR_NAME,
                file_name=file_name,
                cloud_path_prefix=test_cloud_prefix,
            )
            self._write_yolo_config(
                train_dataset_file_name=[],
                val_dataset_file_name=[],
                test_dataset_file_name=file_name,
                root_dir_name=test_root_dir,
                file_name=config_name,
                cloud_path_prefix=test_cloud_prefix,
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
        return DatasetFormat.YOLO_V5_CONFIG
