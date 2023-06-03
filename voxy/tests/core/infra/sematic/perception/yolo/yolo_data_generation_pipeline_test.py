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
from unittest.mock import Mock, patch

from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.perception.yolo.yolo_data_generation_pipeline import (
    yolo_data_generation_pipeline,
)
from core.infra.sematic.shared.mock_sematic_funcs import mock_sematic_funcs
from core.ml.data.generation.common.pipeline import DatasetMetaData
from core.ml.data.generation.resources.api.dataset_generator import (
    generate_and_register_dataset,
)
from core.structs.dataset import DataCollectionLogset, Dataset, DatasetFormat
from core.structs.task import Task


class PipelineTest(unittest.TestCase):
    """Tests for the YOLO Data Generation Pipeline"""

    @patch(
        "core.infra.sematic.perception.yolo.yolo_data_generation_pipeline.generate_logset"
    )
    @patch(
        (
            "core.infra.sematic.perception.yolo.yolo_data_generation_pipeline."
            "get_or_create_task_and_service"
        )
    )
    def test_step_connections(
        self, mock_create_task_and_service: Mock, mock_generate_logset: Mock
    ):
        """Confirm that the pipeline produces no errors and a summary of the ingestion"""
        mock_create_task_and_service.return_value = Task(
            purpose="OBJECT_DETECTION_2D",
            camera_uuids=[],
        )
        mock_generate_logset.return_value = DataCollectionLogset(
            uuid="5b080632-ff2c-4793-93d2-849841097796",
        )
        pipeline_funcs = [
            generate_and_register_dataset,
        ]
        with mock_sematic_funcs(funcs=pipeline_funcs) as func_mocks:
            func_mocks[generate_and_register_dataset].mock.return_value = (
                Dataset(
                    uuid="7d45f01a-7656-4341-a25c-bf3394ffa7fc",
                    data_collection_logset_ref={},
                ),
                DatasetMetaData(
                    cloud_path="s3://voxel-temp/yolo_dataset",
                    config={},
                    dataset_format=DatasetFormat.YOLO_V5_CONFIG,
                    local_path="local_path",
                ),
            )

            # trunk-ignore(pylint/E1101)
            dataset, dataset_metadata = yolo_data_generation_pipeline(
                logset_config=(
                    "core/ml/data/generation/configs/logsets/test/OBJECT_DETECTION_2D.yaml"
                ),
                dataset_config="core/ml/data/generation/configs/datasets/OBJECT_DETECTION_2D.yaml",
                metaverse_environment="INTERNAL",
            ).resolve(SilentResolver())
            self.assertIsInstance(dataset, Dataset)
            self.assertIsInstance(dataset_metadata, DatasetMetaData)
