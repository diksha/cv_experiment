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

import typing
from abc import ABC, abstractmethod

import torch

from core.structs.model import ModelConfiguration


class InferenceBackend(ABC):
    """
    Basic wrapper interface for a backend in triton
    """

    @abstractmethod
    def __init__(self, model_path: str):
        """
        Initializes inference backend

        Args:
            model_path (str): The model name
                        (
                        e.g.: artifacts_02_27_2023_michaels_wesco_office_yolo
                         /best_736_1280.engine.tar.gz
                         )
        """

    @abstractmethod
    def get_config(self) -> ModelConfiguration:
        """
        Returns a dictionary containing the configuration of the model.

        This is related to how to configure preprocessing/post processing
        since that may change from model to model
        (different transforms, etc.)

        Returns:
            ModelConfiguration: A struct containing the configuration of the model.
        """

    @abstractmethod
    def infer(
        self, input_tensors: typing.List[torch.Tensor]
    ) -> typing.List[torch.Tensor]:
        """
        Performs inference on the input tensor x and returns the model
        result

        Args:
            input_tensor (torch.Tensor): The input tensor to perform inference on.

        Returns:
            torch.Tensor: The output tensor of the model that was run on the particular backend

        """
