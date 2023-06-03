import typing

import torch

from core.perception.inference.transforms.registry import get_transforms
from lib.ml.inference.backends.base import InferenceBackend
from lib.ml.inference.provider.interface import AbstractInferenceProvider
from lib.ml.inference.tasks.door_state.vanilla_resnet.utils import (
    extact_door_configs,
    extract_preprocessed_door_image,
    post_process_prediction,
)


class InferenceProvider(AbstractInferenceProvider):
    def __init__(
        self,
        backend: InferenceBackend,
        camera_uuid: str,
        config: typing.Dict[str, object],
        device: torch.device,
        training_config: typing.Dict[str, object],
    ) -> None:
        self.backend = backend
        self.camera_uuid = camera_uuid
        self.device = device
        self.door_configs = None
        # TODO: Get following parameters through InferenceBackend::get_config()
        self.state2class = config["state2class"]
        self.preprocessing_transforms = config[
            "runtime_preprocessing_transforms"
        ]
        self.training_transforms = get_transforms(
            training_config["data"]["transforms"]
        )
        self.postprocessing_transforms = config["postprocessing_transforms"]

    def process(self, batched_input: torch.tensor) -> typing.List[list]:
        """predict.

        Generates a set of predictions by observation class. Format is:
        {
            CategoryType: torch.Tensor,
            ...
        }
        The format of the tensor is [bbox predictions | bbox confidence | class confidences]
        (typical yolo format)

        Args:
            batched_input (torch.tensor): raw NHWC inputs (batch, height, width, channels)

        Returns:
            typing.List[list]: list of door actors in a frame with predictions indexed by frame
        """
        batched_predictions = []
        image_height = batched_input.shape[1]
        image_width = batched_input.shape[2]
        if self.door_configs is None:
            door_configs = extact_door_configs(
                self.camera_uuid, image_height, image_width
            )
        for _, image in enumerate(batched_input.numpy()):
            door_actors = []
            for track_id, door_config in enumerate(door_configs):
                processed_image = extract_preprocessed_door_image(
                    image,
                    door_config,
                    self.preprocessing_transforms,
                    self.training_transforms,
                    self.device,
                )
                door_state_prediction = self.backend.infer([processed_image])[
                    0
                ][0]
                door_actors.append(
                    post_process_prediction(
                        door_state_prediction,
                        self.state2class,
                        track_id,
                        door_config,
                        self.postprocessing_transforms,
                        self.camera_uuid,
                    )
                )
            batched_predictions.append(door_actors)

        return batched_predictions
