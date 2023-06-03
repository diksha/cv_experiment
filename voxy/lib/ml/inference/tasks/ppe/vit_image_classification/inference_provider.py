#
# Copyright 2020-2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

from typing import List

import torch
from transformers import ViTFeatureExtractor

from lib.ml.inference.backends.base import InferenceBackend
from lib.ml.inference.tasks.ppe.vit_image_classification.utils import (
    preprocess_inputs,
)


class InferenceProvider:
    """TorchScript Inference Provider."""

    def __init__(
        self,
        backend: InferenceBackend,
        feature_extractor: "ViTFeatureExtractor",
        padding: int,
        device: torch.device,
    ) -> None:
        """Constructor

        Args:
            jit_file_path (str): Path to torchscript file.
        """
        self.backend = backend
        self.padding = padding
        self.feature_extractor = feature_extractor
        self.device = device

    def process(
        self, actors_xyxy: List[torch.Tensor], image: torch.Tensor
    ) -> torch.Tensor:
        """
        Processes the input frame tensor with the given person actors

        Args:
            actors_xyxy (torch.tensor): list of tensors corresponding to
                people in the image [tl_x, tl_y, bl_x, bl_y]. Must
                have at least one actor!
            image (torch.tensor): the preprocessed model inputs

        Returns:
            torch.tensor: the inferred model outputs

        Raises:
            RuntimeError: empty list passed to for actors_xyxy
        """
        if len(actors_xyxy) == 0:
            raise RuntimeError("Client Error: actors_xyxy should not be empty")

        batched_input = preprocess_inputs(
            actors_xyxy,
            image,
            self.feature_extractor,
            self.padding,
            self.device,
        )
        model_output = self.backend.infer([batched_input])[0]
        # Huggingface returns tuple for jit compiled models
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output
