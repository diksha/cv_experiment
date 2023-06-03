import typing

import torch

from lib.ml.inference.backends.base import InferenceBackend
from lib.ml.inference.provider.interface import AbstractInferenceProvider


class InferenceProvider(AbstractInferenceProvider):
    """TorchScript Inference Provider."""

    def __init__(self, backend: typing.Type[InferenceBackend]) -> None:
        """Constructor

        Args:
            backend (typing.Type[InferenceBackend]): the backend (e.g. torchscript, triton)
                                    to perform inference on
        """
        self.backend = backend

    def process(self, batched_input: torch.Tensor) -> torch.Tensor:
        """
        Processes the input frame tensor with the given person actors

        Args:
            batched_input (torch.Tensor): the preprocessed model inputs

        Returns:
            torch.tensor: the inferred model outputs
        """
        predictions = self.backend.infer([batched_input])[0]
        return predictions
