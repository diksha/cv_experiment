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

from dataclasses import dataclass
from typing import List, Union

from core.structs.incident import Incident


# NOTE: dataclass is used rather than attrs here because Scenario is used with Sematic,
# and Sematic has first-class-citizen support for dataclasses in their visualizations.
@dataclass
class Scenario:
    """A video from an individual camera at a specific time in which >=1 incidents occur

    Attributes
    ----------
    camera_uuid:
        The camera's unique identifier
    incidents:
        A list of "ground truth" incidents occuring within the given video segment
    video_uuid:
        The unique id for the desired video segment
    scenario_for_incidents:
        Scenario pulled for what incident types
    inferred_incidents:
        A list of incidents, determined by the perception model (instead of ground truth).
        When this value is None, it implies that no inference results are known.
    """

    camera_uuid: str
    incidents: List[str]
    video_uuid: str
    scenario_for_incidents: List[str]
    inferred_incidents: Union[List[Incident], List[str], None] = None

    def has_inference_results(self) -> bool:
        """Determine whether inference results are known for this scenario.

        Returns:
            True if the inference results are known, False otherwise.
        """
        return self.inferred_incidents is not None


@dataclass
class ScenarioSet:
    """A named group of related scenarios for running inference on

    Attributes
    ----------
    name:
        The name of the scenario. For example, "door_intersection"
    scenarios:
        The scenarios in the set.
    """

    name: str
    scenarios: List[Scenario]
