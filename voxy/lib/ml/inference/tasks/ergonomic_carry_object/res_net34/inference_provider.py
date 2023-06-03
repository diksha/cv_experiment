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

import torch

from lib.ml.inference.backends.base import InferenceBackend
from lib.ml.inference.provider.interface import AbstractInferenceProvider


class InferenceProvider(AbstractInferenceProvider):
    """Carry Object Inference Provider."""

    def __init__(
        self,
        backend: InferenceBackend,
    ) -> None:
        """Constructor

        Args:
            backend (InferenceBackend): the backend that is able to
                       run inference, e.g. Triton, ONNX, etc.
        """
        self.backend = backend

    def process(self, batched_input: torch.tensor) -> torch.tensor:
        """
        Processes the input frame tensor with the given person actors

        Args:
            batched_input (torch.tensor): the preprocessed model inputs

        Returns:
            torch.tensor: the inferred model outputs
        """
        predictions = self.backend.infer([batched_input])[0]
        return predictions
