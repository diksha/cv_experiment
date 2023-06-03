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

import unittest
from unittest.mock import call, patch

from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.perception.yolo.yolo_options import YoloTrainingOptions
from core.infra.sematic.shared.mock_sematic_funcs import mock_sematic_funcs
from core.ml.data.generation.common.pipeline import DatasetMetaData
from core.structs.dataset import Dataset, DatasetFormat

from core.infra.sematic.perception.yolo.yolo_training_pipeline import (  # isort: skip
    aggregate_val,
    train,
    val,
    yolo_training_pipeline as pipeline,
)


class PipelineTest(unittest.TestCase):
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.VoxelDataset.download"
    )
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.upload_directory_to_s3"
    )
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.download_directory_from_s3"
    )
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.prepare_data_dirs"
    )
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.setup_environment"
    )
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.logged_subprocess_call"
    )
    def test_train_call(
        self,
        logged_subprocess_call_mock,
        setup_environment_mock,
        prepare_data_dirs_mock,
        download_directory_mock,
        upload_directory_mock,
        datset_download_mock,
    ):
        """Test that training process starts"""
        upload_directory_mock.return_value = "s3://path-to-model"
        train(
            Dataset(
                uuid="dataset_uuid",
                data_collection_logset_ref={},
                path="s3://voxel-temp/yolo_dataset",
            ),
            DatasetMetaData(
                cloud_path="s3://voxel-temp/yolo_dataset",
                config={},
                dataset_format=DatasetFormat.YOLO_V5_CONFIG,
                local_path="/tmp/datset",
            ),
            YoloTrainingOptions(
                model_output_bucket="voxel-models",
                model_output_relative_path="yolo",
                weights_path="yolov5m.pt",
                n_epochs=20,
                batch_size=8,
                image_size=1280,
                weights_cfg="yolov5m.yaml",
            ),
        ).resolve(SilentResolver())
        self.assertTrue(not download_directory_mock.called)
        self.assertTrue(datset_download_mock.called)
        setup_environment_mock.assert_called_once_with()
        prepare_data_dirs_mock.assert_called_once_with()
        logged_subprocess_call_mock.assert_called_once()

    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.VoxelDataset.download"
    )
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.upload_directory_to_s3"
    )
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.download_directory_from_s3"
    )
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.prepare_data_dirs"
    )
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.setup_environment"
    )
    @patch(
        "core.infra.sematic.perception.yolo.yolo_training_pipeline.logged_subprocess_call"
    )
    def test_val_call(
        self,
        logged_subprocess_call_mock,
        setup_environment_mock,
        prepare_data_dirs_mock,
        download_directory_mock,
        upload_directory_mock,
        datset_download_mock,
    ):
        """Test that training process starts"""
        upload_directory_mock.return_value = "s3://path-to-val"
        val(
            Dataset(
                uuid="dataset_uuid",
                data_collection_logset_ref={},
                path="s3://voxel-temp/yolo_dataset",
            ),
            DatasetMetaData(
                cloud_path="s3://voxel-temp/yolo_dataset",
                config={},
                dataset_format=DatasetFormat.YOLO_V5_CONFIG,
                local_path="/tmp/datset",
            ),
            YoloTrainingOptions(
                model_output_bucket="voxel-models",
                model_output_relative_path="yolo",
                weights_path="yolov5m.pt",
                n_epochs=20,
                batch_size=8,
                image_size=1280,
                weights_cfg="yolov5m.yaml",
            ),
            model_output="s3://path-to-val",
        ).resolve(SilentResolver())
        download_directory_mock.assert_has_calls(
            [
                call(
                    "/tmp/datset/weights",
                    "voxel-models",
                    "yolo/dataset_uuid/model_output/yolo_automated_dataset_uuid/weights",
                ),
            ],
            any_order=True,
        )
        self.assertTrue(datset_download_mock.called)
        setup_environment_mock.assert_called_once_with()
        prepare_data_dirs_mock.assert_called_once_with()
        logged_subprocess_call_mock.assert_called_once()

    def test_step_connections(self):
        """Test training pipeline step connections"""

        pipeline_funcs = [train, val, aggregate_val]
        with mock_sematic_funcs(funcs=pipeline_funcs) as func_mocks:
            func_mocks[train].mock.return_value = "s3://path-to-model"
            func_mocks[val].mock.return_value = "s3://path-to-val-output"
            func_mocks[
                aggregate_val
            ].mock.return_value = "s3://path-to-aggregate-val-csv"
            # Dataset -> collection of data collection references
            # DatasetMetaData -> cloud info where dataset is stored
            pipeline(
                Dataset(
                    uuid="dataset_uuid",
                    data_collection_logset_ref={},
                    path="s3://voxel-temp/yolo_dataset",
                ),
                DatasetMetaData(
                    cloud_path="s3://voxel-temp/yolo_dataset",
                    config={},
                    dataset_format=DatasetFormat.YOLO_V5_CONFIG,
                    local_path="/tmp/datset",
                ),
                YoloTrainingOptions(
                    model_output_bucket="voxel-models",
                    model_output_relative_path="yolo",
                    weights_path="yolov5m.pt",
                    n_epochs=20,
                    batch_size=8,
                    image_size=1280,
                    weights_cfg="yolov5m.yaml",
                ),
            ).resolve(SilentResolver())
            func_mocks[train].mock.assert_called_once()
            func_mocks[val].mock.assert_called_once()
            func_mocks[aggregate_val].mock.assert_called_once()
