# #
# # Copyright 2020-2021 Voxel Labs, Inc.
# # All rights reserved.
# #
# # This document may not be reproduced, republished, distributed, transmitted,
# # displayed, broadcast or otherwise exploited in any manner without the express
# # prior written permission of Voxel Labs, Inc. The receipt or possession of this
# # document does not convey any rights to reproduce, disclose, or distribute its
# # contents, or to manufacture, use, or sell anything that it may describe, in
# # whole or in part.
# #
import os
import unittest

from mock import patch

# lightly hack to avoid running network call upon import
# https://github.com/lightly-ai/lightly/blob/fc252424afae73af54826d97b36042130522d025/lightly/__init__.py#L126
# trunk-ignore-all(pylint/C0413,flake8/E402)
os.environ["LIGHTLY_DID_VERSION_CHECK"] = "True"
from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.shared.mock_sematic_funcs import mock_sematic_funcs
from core.infra.sematic.shared.utils import PipelineSetup
from core.ml.data.generation.common.pipeline import DatasetMetaData
from core.ml.data.generation.resources.api.dataset_generator import (
    generate_and_register_dataset,
)
from core.ml.experiments.runners.experiment_manager import (
    run_experiment,
    train,
)
from core.ml.training.api.generated_model import GeneratedModel
from core.structs.dataset import DataCollectionLogset
from core.structs.dataset import Dataset as VoxelDataset
from core.structs.dataset import DatasetFormat
from core.structs.task import Task


# To resolve after lightly suggests on ways to mock their api's
class ExperimentManagerTest(unittest.TestCase):
    @patch(
        "core.ml.experiments.runners.experiment_manager.get_secret_from_aws_secret_manager"
    )
    @patch(
        "core.ml.experiments.runners.experiment_manager.get_or_create_task_and_service"
    )
    @patch("core.ml.experiments.runners.experiment_manager.generate_logset")
    @patch("core.ml.experiments.runners.experiment_manager.register_model")
    @patch("core.ml.experiments.runners.experiment_manager.notify_slack")
    @patch("core.ml.experiments.runners.experiment_manager.ExperimentTracker")
    def test_pipeline(
        self,
        experiment_tracker,
        mock_notify_slack,
        mock_register_model,
        mock_generate_logset,
        mock_get_or_create_task_and_service,
        mock_secret_manager,
    ):
        """Testing experiment manager pipeline

        Args:
            experiment_tracker (mock): mock of experiment tracker
            mock_notify_slack (mock): mocking slack notification
            mock_register_model (mock): mocks registering the model
            mock_generate_logset (mock): mock of generate logset
            mock_get_or_create_task_and_service (mock): mock of create task
            mock_secret_manager (mock): mock of secret manager
        """
        pipeline_result = {"metrics": {"recall": 0.9}}
        mock_secret_manager.return_value = "secret"
        mock_get_or_create_task_and_service.return_value = Task(
            camera_uuids=["camera_uuid"],
            service_id=["service_uuid"],
            model_category="IMAGE_CLASSIFICATION",
            purpose="OBJECT_DETECTION_2D",
            uuid="task1",
        )
        mock_generate_logset.return_value = DataCollectionLogset()
        mock_register_model.return_value = pipeline_result
        self.assertTrue(True)  # trunk-ignore(pylint/W1503)
        pipeline_funcs = [generate_and_register_dataset, train]
        with mock_sematic_funcs(funcs=pipeline_funcs) as func_mocks:

            func_mocks[
                generate_and_register_dataset
            ].mock.return_value = VoxelDataset(
                data_collection_logset_ref={}, uuid="dataset_uuid"
            ), DatasetMetaData(
                cloud_path="path",
                config={},
                dataset_format=DatasetFormat.IMAGE_CSV,
                local_path="local_path",
            )
            func_mocks[train].mock.return_value = "s3_path", GeneratedModel(
                local_model_path="local_model_path", metrics={}
            )
            result = run_experiment(
                experiment_config_path="core/ml/experiments/configs/DOOR_STATE.yaml",
                experimenter="buildkite",
                notify=True,
                camera_uuids=["americold/modesto/0011/cha"],
                organization=None,
                location=None,
                metaverse_environment="INTERNAL",
                pipeline_setup=PipelineSetup(),
            ).resolve(SilentResolver())
            self.assertEqual(result, pipeline_result)
