import torch

from lib.ml.inference.backends.base import InferenceBackend
from lib.ml.inference.provider.interface import AbstractInferenceProvider


class InferenceProvider(AbstractInferenceProvider):
    """TorchScript Inference Provider."""

    def __init__(
        self,
        backend: InferenceBackend,
    ) -> None:
        """Constructor

        Args:
            jit_file_path (str): Path to torchscript file.
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
