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

from typing import List, Optional

import sematic
from loguru import logger

from core.infra.sematic.shared.utils import PipelineSetup
from core.metaverse.api.queries import get_or_create_task_and_service
from core.ml.data.flywheel.lib.dataflywheel import (
    DataFlywheel,
    DataFlywheelSummary,
)
from core.structs.model import ModelCategory
from core.structs.task import TaskPurpose


@sematic.func
# trunk-ignore(pylint/R0913)
def run_dataflywheel(
    task_purpose: str,
    model_category: str,
    camera_uuids: List[str],
    should_notify: bool,
    start_date: Optional[str],
    end_date: Optional[str],
    max_incidents: int,
    metaverse_environment: str,
    overwrite_config_file: Optional[str],
    pipeline_setup: PipelineSetup,
) -> DataFlywheelSummary:
    """Initialize and run the dataflywheel
    Args:
        task_purpose (str): task purpose to use for dataflywheel
        model_category (str): model category
        camera_uuids (List[str]): list of camera uuids to initialize the task
        should_notify (bool): notify slack flag
        metaverse_environment (str): metavrse environment to run
        max_incidents: (int): max number of incidents to query from portal
        start_date (Optional[str]): start date to query incidents from portal
        end_date (Optional[str]): end date to query incidents from portal
        overwrite_config_file (Optional[str]): config that will overwrite dataflywheel configs
        pipeline_setup (PipelineSetup): setting up some defaults for pipeline
    Returns:
        DataFlywheelSummary: summary of dataflywheel run
    """
    logger.info(
        (
            "run_dataflywheel," f"task_purpose,{task_purpose},",
            f"model_category,{model_category},"
            f"should_notify,{should_notify},"
            f"metaverse_environment,{metaverse_environment},"
            f"start_date,{start_date},"
            f"end_date,{end_date},"
            f"camera_uuids,{camera_uuids}",
        )
    )
    task = get_or_create_task_and_service(
        TaskPurpose[task_purpose],
        ModelCategory[model_category],
        camera_uuids,
        metaverse_environment=metaverse_environment,
    )
    return DataFlywheel(
        task=task,
        notify=should_notify,
        start_date=start_date,
        end_date=end_date,
        max_incidents=max_incidents,
        overwrite_config_file=overwrite_config_file,
    ).run(metaverse_environment=metaverse_environment)
