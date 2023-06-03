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

import json
import os
from dataclasses import dataclass
from typing import List

from jsondiff import diff
from loguru import logger


@dataclass
class TaxonomyInfo:
    """Information about taxonomy"""

    parent_taxonomy_project: str
    attribute_info: List[str]


taxonomy_map = {
    "safety_vest_image_annotation": TaxonomyInfo(
        "video_playback_annotation", ["safety_vest", "bare_chest"]
    ),
    "safety_gloves_image_annotation": TaxonomyInfo(
        "video_playback_annotation", ["safety_glove", "bare_hand"]
    ),
}


def validate_taxonomy(project: str) -> bool:
    """Validate the taxonomy in scale

    Args:
        project (str): Project to validate

    Returns:
        bool: if taxonomy is valid
    """
    if project not in taxonomy_map:
        logger.info(f"No parent taxonomy for {project}")
        return True
    taxonomy_parent = "core/labeling/scale/task_creation/taxonomies"
    taxonomy_path = os.path.join(taxonomy_parent, f"{project}.json")
    to_diff_taxonomy_path = os.path.join(
        taxonomy_parent,
        f"{taxonomy_map[project].parent_taxonomy_project}.json",
    )
    with open(taxonomy_path, "r", encoding="UTF-8") as taxonomy_file:
        taxonomy_1 = json.load(taxonomy_file)
    with open(
        to_diff_taxonomy_path, "r", encoding="UTF-8"
    ) as to_diff_taxonomy_file:
        taxonomy_2 = json.load(to_diff_taxonomy_file)

    taxonomy_diff = diff(taxonomy_1, taxonomy_2)
    for attribute_info in taxonomy_map[project].attribute_info:
        if attribute_info in str(taxonomy_diff).lower():
            logger.warning(
                f"Diff between parent and child taxonomies {taxonomy_diff}"
            )
            return False

    return True
