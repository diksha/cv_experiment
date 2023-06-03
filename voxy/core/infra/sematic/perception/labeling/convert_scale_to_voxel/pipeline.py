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

from typing import Optional

import sematic
from loguru import logger

from core.infra.sematic.shared.utils import PipelineSetup
from core.labeling.label_conversion.add_to_metaverse import (
    add_consumable_label_batch_to_metaverse,
)
from core.labeling.scale.lib.converters.scale_voxel_conversion import (
    convert_scale_to_voxel_sematic_wrapper,
)
from core.ml.data.validation.lib.class_distribution import (
    notify_class_distribution,
)


@sematic.func
def pipeline(
    completion_before_date: str,
    lookback_days: float,
    converter_type: str,
    consumable_labels_fn: Optional[str],
    project_name: str,
    credentials_arn: str,
    pipeline_setup: PipelineSetup,
    metaverse_environment: str = "INTERNAL",
) -> bool:
    """
    The root function of the pipeline.

    Args:
        completion_before_date (str): date tasks must be completed before in order
            to be converted, format %Y-%m-%d %H:%M:%S
        lookback_days (float): number of days to look back from completion_before_date
            to convert tasks
        converter_type (str): Converter registered in the ScaleLabelConverterRegistry
        consumable_labels_fn (str): key for conversion function for scale to voxel conversion
        project_name (str): project name in scale to search for tasks
        metaverse_environment (str): metaverse environment
        credentials_arn (str): aws arn containing scale credentials
        pipeline_setup (PipelineSetup): setting up some defaults for pipeline
    Returns:
        bool: pipeline success status
    """

    logger.info(
        (
            "Entering scale to voxel conversion pipeline,"
            f"completion_before_date,{completion_before_date},"
            f"lookback_days,{lookback_days},"
            f"converter_type,{converter_type},"
            f"consumable_labels_fn,{consumable_labels_fn},"
            f"project_name,{project_name},"
            f"metaverse_environment,{metaverse_environment}"
        )
    )
    videos_data = convert_scale_to_voxel_sematic_wrapper(
        completion_before_date,
        lookback_days,
        converter_type,
        consumable_labels_fn,
        project_name,
        credentials_arn,
    )
    successful_videos = add_consumable_label_batch_to_metaverse(
        videos_data,
        metaverse_environment,
        "scale",
        project_name,
    )
    notified_status = notify_class_distribution(
        successful_videos, metaverse_environment, project_name
    )
    return notified_status
