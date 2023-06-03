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
from abc import ABC, abstractmethod

from core.ml.experiments.tracking.experiment_tracking import ExperimentTracker
from core.ml.training.models.model_training_result import ModelTrainingResult
from core.structs.dataset import Dataset as VoxelDataset

# Please remove as soon as there is time available
# trunk-ignore-all(pylint/C0116)
# trunk-ignore-all(pylint/C0115)


class Model(ABC):
    """
    Utility class to define how training a model happens.

    Clients must implement the `train()` and `evaluate()` methods
    """

    def __init__(self, config: dict):
        pass

    @abstractmethod
    def train(
        self, dataset: VoxelDataset, experiment_tracker: ExperimentTracker
    ) -> ModelTrainingResult:
        raise NotImplementedError("To be implemented in derived class")

    @classmethod
    @abstractmethod
    def evaluate(
        cls,
        model_trainging_context: ModelTrainingResult,
        dataset: VoxelDataset,
        experiment_tracker: ExperimentTracker,
    ):
        raise NotImplementedError("To be implemented in derived class")
