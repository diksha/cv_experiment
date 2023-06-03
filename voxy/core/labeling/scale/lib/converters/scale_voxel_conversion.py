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
from datetime import datetime, timedelta
from typing import List, Optional

import sematic

import core.labeling.scale.registry.register_components  # trunk-ignore(pylint/W0611,flake8/F401)
from core.infra.sematic.shared.resources import CPU_1CORE_4GB
from core.labeling.scale.lib.converters.converter_base import VideoData
from core.labeling.scale.lib.converters.door_actor_utils import (
    generate_consumable_labels_for_doors,
)
from core.labeling.scale.lib.converters.ppe_hat_actor_utils import (
    generate_consumable_labels_for_ppe_hat,
)
from core.labeling.scale.lib.converters.safety_vest_actor_utils import (
    generate_consumable_labels_for_safety_vest,
)
from core.labeling.scale.registry.registry import ScaleLabelConverterRegistry

# TODO: PERCEPTION-1907
consumable_labels_fn_map = {
    "generate_consumable_labels_for_doors": generate_consumable_labels_for_doors,
    "generate_consumable_labels_for_safety_vest": generate_consumable_labels_for_safety_vest,
    "generate_consumable_labels_for_ppe_hat": generate_consumable_labels_for_ppe_hat,
}


def convert_scale_to_voxel(
    completion_before_date: str,
    lookback_days: float,
    converter_type: str,
    consumable_labels_fn: str,
    project_name: str,
    credentials_arn: str,
) -> List[VideoData]:
    """
    Converts scale labels to voxel format and uploads to cloud
    Args:
        completion_before_date (str): date tasks must be completed before in order
            to be converted, format %Y-%m-%d %H:%M:%S
        lookback_days (float): number of days to look back from completion_before_date
            to convert tasks
        converter_type (str): Converter registered in the ScaleLabelConverterRegistry
        consumable_labels_fn (str): key for conversion function for scale to voxel conversion
        project_name (str): project name in scale to search for tasks
        credentials_arn (str): arn of scale credentials
    Returns:
        List[str]: list of video uuids successfully converted and put in the cloud
    """
    end_date = datetime.strptime(completion_before_date, "%Y-%m-%d %H:%M:%S")
    duration = timedelta(days=lookback_days)
    completion_after_date = (end_date - duration).strftime("%Y-%m-%d %H:%M:%S")
    scale_converter = ScaleLabelConverterRegistry.get_instance(
        converter_type,
        {
            "completion_before_date": completion_before_date,
            "completion_after_date": completion_after_date,
            "consumable_labels_fn": consumable_labels_fn_map.get(
                consumable_labels_fn
            ),
            "project_name": project_name,
            "credentials_arn": credentials_arn,
        },
    )
    return scale_converter.convert_and_upload()


@sematic.func(resource_requirements=CPU_1CORE_4GB, standalone=True)
def convert_scale_to_voxel_sematic_wrapper(
    completion_before_date: str,
    lookback_days: float,
    converter_type: str,
    consumable_labels_fn: Optional[str],
    project_name: str,
    credentials_arn: str,
) -> List[VideoData]:
    """
    Sematic wrapper for convert_scale_to_voxel
    Args:
        completion_before_date (str): date tasks must be completed before in order
            to be converted, format %Y-%m-%d %H:%M:%S
        lookback_days (float): number of days to look back from completion_before_date
            to convert tasks
        converter_type (str): Converter registered in the ScaleLabelConverterRegistry
        consumable_labels_fn (str): key for conversion function for scale to voxel conversion
        project_name (str): project name in scale to search for tasks
        credentials_arn (str): aws arn containing scale credentials
    Returns:
        List[str]: list of video uuids successfully converted and put in the cloud
    """
    return convert_scale_to_voxel(
        completion_before_date,
        lookback_days,
        converter_type,
        consumable_labels_fn,
        project_name,
        credentials_arn,
    )
