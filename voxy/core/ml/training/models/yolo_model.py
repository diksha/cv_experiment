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

from core.ml.training.models.model import Model
from core.ml.training.registry.registry import ModelClassRegistry

# docstrings:
# trunk-ignore-all(pylint/C0116)


@ModelClassRegistry.register()
class YoloModel(Model):
    """Defines how the training of yolo happens."""

    def __init__(self, model_metadata, dataset, experiment_tracker):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def get_model_metadata(self) -> dict:
        """Get model metadata

        Returns:
            dict: metadata information
        """
        raise NotImplementedError("To be implemented")
