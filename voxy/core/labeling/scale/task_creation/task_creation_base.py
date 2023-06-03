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

import hashlib
import json
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from loguru import logger

from core.labeling.scale.lib.scale_client import get_scale_client
from core.labeling.scale.lib.utils import validate_taxonomy
from core.structs.data_collection import DataCollectionType


class TaskCreation(ABC):

    _TAXONOMY_PATH = "core/labeling/scale/task_creation/taxonomies"

    def __init__(
        self,
        project,
        credentials_arn: str,
        callback_url=None,
        batch_name_prefix="",
        dry_run=False,
    ):
        self.project = project
        if not project:
            raise RuntimeError("Project name should be set")
        self.client = get_scale_client(credentials_arn)
        self.credentials_arn = credentials_arn
        current_time = datetime.strftime(datetime.now(), "%d_%m_%Y_%H_%M")
        taxonomy_path = os.path.join(self._TAXONOMY_PATH, f"{project}.json")
        with open(taxonomy_path, "r", encoding="UTF-8") as taxonomy_file:
            self.taxonomy = json.load(taxonomy_file)
        if not validate_taxonomy(project):
            raise RuntimeError(f"Taxonomy not valid for {project}")

        self.batch = self.client.create_batch(
            project=project,
            batch_name=f"batch_{batch_name_prefix}_{current_time}"
            if batch_name_prefix
            else f"batch_{current_time}",
            callback=callback_url,
        )
        self.dry_run = dry_run
        self.uuid = str(uuid.uuid4())
        logger.info(f"Created scale batch {self.project} {self.batch.name}")

    def get_taxonomy_version(self, taxonomy: dict) -> str:
        """Given the taxonomy file, creates a sha

        Args:
            taxonomy (dict): Taxonomy from file for project

        Returns:
            str: sha of the taxonomy dictionary
        """
        return hashlib.sha256(
            json.dumps(taxonomy, sort_keys=True).encode("utf-8")
        ).hexdigest()

    @abstractmethod
    def create_task(
        self, video: str, fps: float, generate_hypothesis: bool = False
    ) -> List[str]:
        """Create one or more Scale tasks and return unique ids for each created task

        Args:
             video: UUID for the Video tasks are being created for
             fps: frames-per-second of the video to create labelling tasks for
             generate_hypothesis: whether to generate hypothesis for the task

        Returns:
            A list of unique ids to identify each task created
        """
        raise NotImplementedError(
            "TaskCreation must implement create_task method"
        )

    @abstractmethod
    def get_data_collection_type(self) -> DataCollectionType:
        """Get DataCollectionType for task creator"""
        raise NotImplementedError(
            "TaskCreation must implement create_task method"
        )

    def finalize(self):
        self.batch.finalize()
        logger.info(f"Batch for project {self.project} {self.batch.name}")
