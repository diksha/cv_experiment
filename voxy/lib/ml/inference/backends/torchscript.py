import typing

import torch

from core.structs.model import ModelConfiguration
from lib.ml.inference.backends.base import InferenceBackend


class TorchscriptBackend(InferenceBackend):
    """TorchscriptBackend."""

    def __init__(
        self,
        jit_file_path: str,
        device: torch.device,
        dtypes: typing.List[torch.dtype],
        **kwargs: dict,
    ) -> None:
        """Consturctor.

        Args:
            jit_file_path (str): Path to torchscript file.
            device (torch.device): Device whether cuda 0, cpu etc.
            dtype (torch.dtype): Inference precision type half or float
        Raises:
            RuntimeError: Unsupported dtype.
        """
        self.model = torch.jit.load(jit_file_path, **kwargs).eval()
        self.device = device
        self.dtypes = dtypes

        if any(dtype is torch.half for dtype in dtypes):
            self.model = self.model.half()
        elif any(dtype is torch.float for dtype in dtypes):
            self.model = self.model.float()
        else:
            raise RuntimeError(f"Unsupported dtype collection {self.dtypes}!")

        self.model = self.model.to(device)

    def get_config(self) -> ModelConfiguration:
        """
        Loads a model configuration relating to all relevant preprocessing
        postprocessing for the model

        This is related to how to configure preprocessing/post processing
        since that may change from model to model
        (different transforms, etc.)

        Raises:
           RuntimeError: As the method is not implemented
        """
        # TODO: update this to grab the proper model configuration
        raise RuntimeError("Not Implemented!")

    def infer(
        self, input_tensors: typing.List[torch.Tensor]
    ) -> typing.List[torch.Tensor]:
        """Infer function that takes an input runs the model on it and return output predictions.

        Args:
            input_tensors (torch.Tensor): input to the model, in case of image generally NCHW.

        Returns:
            torch.Tensor: Ouputs generate by the model inference.
        """
        with torch.no_grad():
            input_tensors_typed = [
                input_tensor.type(dtype).to(self.device)
                for input_tensor, dtype in zip(input_tensors, self.dtypes)
            ]
            return [self.model(*input_tensors_typed)]
