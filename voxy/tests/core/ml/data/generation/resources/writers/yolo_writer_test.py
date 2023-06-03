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

import glob
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from core.ml.data.generation.resources.streams.synchronized_readers import (
    DataCollectionMetaData,
)
from core.ml.data.generation.resources.writers.yolo_writer import (
    YOLOCollector,
    YOLOWriter,
)
from core.utils.yaml_jinja import items_from_file, load_yaml_with_jinja


class TestYOLOWriter(unittest.TestCase):
    """Tests YOLOWriter utility methods"""

    def test_get_name(self) -> None:
        """Tests if the name that is written is formated properly"""
        metadata = DataCollectionMetaData(
            source_name="americold/modesto/0001/cha/test_video",
            index=100,
            time_ms=230000,
        )
        image_name = YOLOCollector.get_name(metadata)
        self.assertEqual(
            image_name, "americold_modesto_0001_cha_test_video_frame_230000"
        )

    @patch("core.ml.data.generation.resources.writers.yolo_writer.upload_file")
    def test_write_dataset(self, mock_s3_upload: MagicMock) -> None:
        """Test write dataset
        Args:
            mock_s3_upload (MagicMock): mocked upload_file
        """
        mock_s3_upload.return_value = ""
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = YOLOWriter(
                actor_categories=[
                    "PIT_V2",
                    "PERSON_V2",
                ],
                output_directory=tmp_dir,
                output_cloud_bucket="voxel-temp",
                output_cloud_prefix="yolo_writer/test",
            )
            collector = writer.create_collector()
            train_images = []
            for i in range(6):
                metadata1 = DataCollectionMetaData(
                    source_name="americold/modesto/0001/cha/video_name",
                    index=i,
                    time_ms=i * 1000,
                )
                metadata2 = DataCollectionMetaData(
                    source_name="americold/tacoma/0001/cha/video_name",
                    index=i,
                    time_ms=i * 1000,
                )
                train_images.append(YOLOCollector.get_name(metadata1))
                train_images.append(YOLOCollector.get_name(metadata2))
                for j in range(5):
                    collector.collect_and_upload_data(
                        data=np.random.randn(480, 960, 3),
                        label=f"{j} 0 0 0 0",
                        is_test=False,
                        metadata=metadata1,
                    )
                    collector.collect_and_upload_data(
                        data=np.random.randn(480, 960, 3),
                        label=f"{j} 0 0 0 0",
                        is_test=False,
                        metadata=metadata2,
                    )
            test_images = []
            for i in range(6):
                metadata1 = DataCollectionMetaData(
                    source_name="americold/ontario/0001/cha/video_name",
                    index=i,
                    time_ms=i * 1000,
                )
                metadata2 = DataCollectionMetaData(
                    source_name="americold/sanford/0001/cha/video_name",
                    index=i,
                    time_ms=i * 1000,
                )
                test_images.append(YOLOCollector.get_name(metadata1))
                test_images.append(YOLOCollector.get_name(metadata2))
                for j in range(3):
                    collector.collect_and_upload_data(
                        data=np.random.randn(480, 960, 3),
                        label=f"{i} 0 0 0 0",
                        is_test=True,
                        metadata=metadata1,
                    )
                    collector.collect_and_upload_data(
                        data=np.random.randn(480, 960, 3),
                        label=f"{j} 0 0 0 0",
                        is_test=True,
                        metadata=metadata2,
                    )
            data_collections = [collector.dump()]
            writer.write_dataset(data_collections)

            # We should have train.txt, val.txt, and the yolo.yaml
            train_datasets = glob.glob(f"{tmp_dir}/train/configs/*")
            self.assertEqual(len(train_datasets), 3)
            # We should have 12 images + labels (6 per camera, 2 cameras)
            train_images = glob.glob(f"{tmp_dir}/train/images/*.png")
            train_labels = glob.glob(f"{tmp_dir}/train/labels/*.txt")
            self.assertEqual(len(train_images), 12)
            self.assertEqual(len(train_labels), 12)
            # We should have test.txt, and the yolo.yaml, for the entire
            # test set, the americold/modesto AND americold/tacomoa train, val
            # sets, and the americold/ontario AND americold/sanford test sets
            # Which total to 14 files
            test_datasets = glob.glob(f"{tmp_dir}/test/configs/*")
            self.assertEqual(len(test_datasets), 14)
            # We should have 10 images + labels (6 per camera, 2 cameras)
            test_images = glob.glob(f"{tmp_dir}/test/images/*.png")
            test_labels = glob.glob(f"{tmp_dir}/test/labels/*.txt")
            self.assertEqual(len(test_images), 12)
            self.assertEqual(len(test_labels), 12)
            # Open config for training, verify paths
            training_config = load_yaml_with_jinja(
                f"{tmp_dir}/train/configs/training_config.yaml"
            )
            self.assertTrue(
                training_config["names"] == ["PERSON_V2", "PIT_V2"]
            )
            self.assertEqual(training_config["nc"], 2)
            self.assertEqual(
                training_config["path"], f"{tmp_dir}/train/configs"
            )
            self.assertEqual(training_config["train"], "training_data.txt")
            self.assertEqual(training_config["val"], "validation_data.txt")
            self.assertTrue(training_config["test"] == [])
            # Open training dataset, verify paths
            training_data = items_from_file(
                f"{tmp_dir}/train/configs/training_data.txt"
            )
            count_modesto = 0
            count_tacoma = 0
            for file_name in training_data:
                self.assertTrue(f"{tmp_dir}/train/images" in file_name)
                count_modesto += "americold_modesto" in file_name
                count_tacoma += "americold_tacoma" in file_name
            # Assert 80% modesto and tacoma data in training text file
            self.assertEqual(count_modesto, 4)
            self.assertEqual(count_tacoma, 4)
