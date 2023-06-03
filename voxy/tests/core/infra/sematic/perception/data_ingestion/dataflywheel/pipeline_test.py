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

# lightly hack to avoid running network call upon import
# https://github.com/lightly-ai/lightly/blob/fc252424afae73af54826d97b36042130522d025/lightly/__init__.py#L126
# trunk-ignore-all(pylint/C0413,flake8/E402)
os.environ["LIGHTLY_DID_VERSION_CHECK"] = "True"
import unittest
from unittest.mock import Mock, patch

from sematic.resolvers.silent_resolver import SilentResolver

from core.infra.sematic.perception.data_ingestion.dataflywheel.pipeline import (
    run_dataflywheel,
)
from core.infra.sematic.shared.mock_sematic_funcs import mock_sematic_funcs
from core.infra.sematic.shared.utils import PipelineSetup
from core.labeling.scale.runners.create_scale_tasks import (
    ScaleTaskSummary,
    create_scale_tasks,
)
from core.ml.data.collection.data_collector import (
    Incident,
    select_incidents_from_portal,
)
from core.ml.data.curation.lib.lightly_worker import (
    LightlyVideoFrameSequence,
    run_lightly_worker,
)
from core.ml.data.flywheel.lib.dataflywheel import (
    DataCollectionInput,
    DataFlywheelSummary,
    generate_data_collections,
    generate_flywheel_summary_and_update_datapool,
    ingest_data_collection_from_collection_input,
)
from core.structs.data_collection import DataCollectionType
from core.structs.datapool import Datapool
from core.structs.frame import Frame
from core.structs.task import Task


class PipelineTest(unittest.TestCase):
    """Unit tests for dataflywheel sematic pipeline"""

    def setUp(self) -> None:
        """Setup ivars for testing"""
        self.incidents = [
            Incident(
                camera_uuid="",
                video_s3_path="",
                incident_type_id="",
                incident_uuid="",
                organization="",
                zone="",
                experimental=True,
                feedback_type="",
                scenario_type="",
            ),
        ]
        self.lightly_output = [
            LightlyVideoFrameSequence(
                video_name="",
                frame_names=[""],
                frame_indices=[0],
                frame_timestamps_pts=[0],
                frame_timestamps_sec=[0],
            )
        ]
        self.data_collection_input = [
            DataCollectionInput(
                data_collection_name="",
                camera_uuid="",
                frames=[
                    Frame(
                        frame_number=0,
                        frame_width=10,
                        frame_height=10,
                        relative_timestamp_s=0,
                        relative_timestamp_ms=0,
                        epoch_timestamp_ms=0,
                    )
                ],
                data_dir="",
                is_test=False,
            ),
        ]
        self.ingested_data_collections = [""]
        self.scale_task_summary = ScaleTaskSummary(
            video_uuids=[""],
            fps=0,
            task_type="VideoPlaybackAnnotationTask",
            data_type=DataCollectionType.VIDEO,
            batch_name="",
            task_unique_ids=[""],
            generate_hypothesis=False,
        )
        self.flywheel_summary = DataFlywheelSummary(
            scale_task_summary=self.scale_task_summary,
            did_update_datapool=True,
        )

    @patch(
        "core.ml.data.flywheel.lib.dataflywheel.get_or_create_datapool_from_task"
    )
    @patch(
        (
            "core.infra.sematic.perception.data_ingestion."
            "dataflywheel.pipeline.get_or_create_task_and_service"
        )
    )
    def test_setup_connections(
        self, mock_task_query: Mock, mock_datapool_query: Mock
    ):
        """Confirms pipeline produces no errors and the flywheel summary
        Args:
            mock_task_query (Mock): task query mock
            mock_datapool_query (Mock): datapool query mock
        """
        mock_task_query.return_value = Task(
            purpose="DOOR_STATE",
            camera_uuids=["americold/modesto/0011/cha"],
        )
        mock_datapool_query.return_value = Datapool(
            input_directory="input_directory",
            uuid="uuid",
            lightly_uuid="lightly_uuid",
            name="name",
            output_directory="output_directory",
            dataset_type="dataset_type",
            ingested_data_collections=[],
        )
        pipeline_funcs = [
            select_incidents_from_portal,
            run_lightly_worker,
            generate_data_collections,
            ingest_data_collection_from_collection_input,
            create_scale_tasks,
            generate_flywheel_summary_and_update_datapool,
        ]
        with mock_sematic_funcs(funcs=pipeline_funcs) as func_mocks:
            func_mocks[
                select_incidents_from_portal
            ].mock.return_value = self.incidents
            func_mocks[
                run_lightly_worker
            ].mock.return_value = self.lightly_output
            func_mocks[
                generate_data_collections
            ].mock.return_value = self.data_collection_input
            func_mocks[
                ingest_data_collection_from_collection_input
            ].mock.return_value = self.ingested_data_collections
            func_mocks[
                create_scale_tasks
            ].mock.return_value = self.scale_task_summary
            func_mocks[
                generate_flywheel_summary_and_update_datapool
            ].mock.return_value = self.flywheel_summary
            # trunk-ignore(pylint/E1101)
            result: DataFlywheelSummary = run_dataflywheel(
                task_purpose="DOOR_STATE",
                model_category="IMAGE_CLASSIFICATION",
                camera_uuids=["americold/modesto/0011/cha"],
                should_notify=False,
                start_date=None,
                end_date=None,
                max_incidents=1,
                metaverse_environment="INTERNAL",
                overwrite_config_file=None,
                pipeline_setup=PipelineSetup(),
            ).resolve(SilentResolver())
            self.assertIsInstance(result, DataFlywheelSummary)
