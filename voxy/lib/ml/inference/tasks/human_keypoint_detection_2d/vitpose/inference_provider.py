import typing

import torch

from lib.ml.inference.backends.base import InferenceBackend
from lib.ml.inference.tasks.human_keypoint_detection_2d.vitpose.utils import (
    post_process,
    preprocess_frame,
)


class InferenceProvider:
    """TorchScript Inference Provider."""

    def __init__(
        self,
        backend: InferenceBackend,
        padding: int,
        device: torch.device,
    ) -> None:
        """Constructor

        Args:
            jit_file_path (str): Path to torchscript file.
        """
        self.backend = backend
        self.padding = padding
        self.device = device
        self.image_size = [192, 256]

    def process(
        self, actors_xyxy: typing.List[torch.Tensor], image: torch.Tensor
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

        (batched_input, original_sizes), origins_xy = preprocess_frame(
            actors_xyxy,
            image,
            self.padding,
            self.image_size,
            self.device,
        )
        model_output = self.backend.infer([batched_input])[0]
        return post_process(model_output, original_sizes, origins_xy)
