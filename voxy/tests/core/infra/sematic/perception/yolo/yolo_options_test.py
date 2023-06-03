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

import argparse
import unittest

from core.infra.sematic.perception.yolo.yolo_options import (
    DataSplit,
    YoloTrainingOptions,
    get_yolo_config_name,
    get_yolo_dataset_name,
)


class YoloNameTest(unittest.TestCase):
    def test_get_yolo_dataset_name(self):
        """Tests whether we get the expeted yolo dataset naming convention"""
        self.assertEqual(
            get_yolo_dataset_name(DataSplit.TRAINING), "training_data.txt"
        )
        self.assertEqual(
            get_yolo_dataset_name(DataSplit.VALIDATION), "validation_data.txt"
        )
        self.assertEqual(
            get_yolo_dataset_name(DataSplit.TESTING), "testing_data.txt"
        )
        self.assertEqual(
            get_yolo_dataset_name(DataSplit.TESTING), "testing_data.txt"
        )
        self.assertEqual(
            get_yolo_dataset_name(DataSplit.TESTING, "test_prefix_"),
            "test_prefix_testing_data.txt",
        )

    def test_get_yolo_config_name(self):
        """Tests whether we get the expeted yolo dataset naming convention"""
        self.assertEqual(
            get_yolo_config_name(DataSplit.TRAINING), "training_config.yaml"
        )
        self.assertEqual(
            get_yolo_config_name(DataSplit.VALIDATION),
            "validation_config.yaml",
        )
        self.assertEqual(
            get_yolo_config_name(DataSplit.TESTING), "testing_config.yaml"
        )
        self.assertEqual(
            get_yolo_config_name(DataSplit.TESTING), "testing_config.yaml"
        )
        self.assertEqual(
            get_yolo_config_name(DataSplit.TESTING, "test_prefix_"),
            "test_prefix_testing_config.yaml",
        )


class YoloTrainingOptionsTest(unittest.TestCase):
    def test_option_parser(self):
        """Tests that option parse can add and parse options"""
        parser = argparse.ArgumentParser()
        YoloTrainingOptions.add_to_parser(parser)
        args = parser.parse_args([])
        obj = YoloTrainingOptions.from_args(args)
        self.assertIsInstance(obj, YoloTrainingOptions)
