#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

from typing import Dict

from core.ml.data.generation.common.registry import LogsetGeneratorRegistry
from core.ml.data.generation.resources.api import (  # trunk-ignore(flake8/F401,pylint/W0611)
    register_components,
)
from core.structs.dataset import DataCollectionLogset
from core.structs.task import Task
from core.utils.yaml_jinja import load_yaml_with_jinja


def load_logset_config(config_file: str, task: Task) -> Dict[str, object]:
    """
    Loads the logset configuration file

    Args:
        config_file (str): the file to load the logset config
        task (Task): the task to help generate the logset

    Returns:
        Dict[str, object]: the loaded config dictionary
    """
    return load_yaml_with_jinja(config_file, task=task.to_dict())


def generate_logset(config: dict) -> DataCollectionLogset:
    """
    Generates a logset given a dataset generation config

    Args:
        config (dict): the top level logset generation config

    Returns:
        DataCollectionLogset: the generated logset
    """
    logset_generator = LogsetGeneratorRegistry.get_instance(
        **config["logset_generation"]
    )
    return logset_generator.generate_logset()
