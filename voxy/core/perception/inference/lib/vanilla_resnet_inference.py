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
import torch
from PIL import Image
from torch import nn

from core.perception.inference.lib.inference import InferenceModel
from core.perception.inference.transforms.registry import get_transforms


class VanillaResnetInferenceModel(InferenceModel):
    def __init__(self, model: nn.Module, config: dict):
        self.device = torch.device(
            config["device"] if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.config = config
        self.transforms = get_transforms(self.config["data"]["transforms"])

    def _preprocess(self, frame: Image) -> torch.Tensor:
        """Preprocess the frame

        Args:
            frame (Image): Image to preprocess

        Returns:
            torch.Tensor: tensor with preprocessed input
        """

        return self.transforms(frame).to(self.device).unsqueeze(0)

    def _postprocess(self, predictions: list) -> torch.Tensor:
        """Post process output after inference

        Args:
            predictions (list): predictions from infer

        Returns:
            torch.Tensor: Tensor with post processed output
        """
        return torch.softmax(predictions[0], dim=0).cpu()

    def infer(self, frame: Image) -> torch.Tensor:
        """Predictions for a given model given image.

        Note: PIL Image in RGB Format is to be passed here.

        Args:
            frame (Image): PIL Image of RGB Format

        Returns:
            torch.Tensor: Predictions as tensor
        """

        preprocessed_input = self._preprocess(frame)
        predictions = self.model(preprocessed_input)
        return self._postprocess(predictions)
