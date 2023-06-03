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
from pathlib import Path

from core.structs.scenario import Scenario, ScenarioSet
from core.utils.yaml_jinja import load_yaml_with_jinja


def from_configuration_file(configuration_file_path: str) -> ScenarioSet:
    """Load a set of scenarios from one of the files in data/scenario_sets/regression

    Args:
        configuration_file_path: The path to the file defining the configurations to be executed

    Returns:
        A ScenarioSet object parsed from the file.

    Raises:
        ValueError: Raises if the configuration file's contents are invalid.
    """
    file_name = Path(configuration_file_path).parts[-1]
    scenario_set_name = file_name.replace(".yaml", "")
    config_dict = load_yaml_with_jinja(configuration_file_path)
    if "scenarios" not in config_dict:
        raise ValueError(
            f"The yaml from '{configuration_file_path}' doesn't contain a valid "
            f"configuration for a set of scenarios. The contained yaml has no "
            f"'scenarios' key."
        )

    if "scenario_for_incidents" in config_dict:
        for scenario in config_dict["scenarios"]:
            scenario["scenario_for_incidents"] = config_dict[
                "scenario_for_incidents"
            ]

    scenarios = [Scenario(**sc) for sc in config_dict["scenarios"]]

    return ScenarioSet(
        name=scenario_set_name,
        scenarios=scenarios,
    )
