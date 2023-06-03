import json

import torch

from lib.ml.inference.backends.torchscript import TorchscriptBackend
from lib.ml.inference.backends.triton import TritonInferenceBackend
from lib.ml.inference.factory.base import (
    InferenceBackendType,
    InferenceProviderFactoryBase,
)
from lib.ml.inference.tasks.door_state.vanilla_resnet.inference_provider import (
    InferenceProvider,
)


class InferenceProviderFactory(InferenceProviderFactoryBase):
    """Inference Provider Factory."""

    def get_inference_provider(
        self,
        model_path: str,
        camera_uuid: str,
        config: dict,
    ) -> InferenceProvider:
        """Get correct vanilla resnet inference provider class
        Args:
            model_path (str): path to model
            camera_uuid (str): camera uuid running perception system
            config (dict): config for running inference containing
                states, additional preprocessing and postprocessing inputs
        Returns:
            InferenceProvider: Vanilla resnet inference provider
        Raises:
            RuntimeError: type not valid inference provider type
        """
        extra_files = {"model_config": ""}

        def generate_torchscript_inference_backend() -> TorchscriptBackend:
            """Generate torchscript inference backend for door state
            Returns:
                TorchscriptBackend
            """
            return TorchscriptBackend(
                jit_file_path=model_path,
                device=torch.device(self._device),
                dtypes=[torch.float],
                _extra_files=extra_files,
            )

        def generate_triton_inference_backend() -> TritonInferenceBackend:
            """Generate triton inference backend for door state
            Returns:
                TritonInferenceBackend
            """
            torch.jit.load(
                model_path, _extra_files=extra_files, map_location="cpu"
            )
            return TritonInferenceBackend(
                model_name=model_path, triton_server_url=self.triton_server_url
            )

        type_map = {
            InferenceBackendType.TORCHSCRIPT: generate_torchscript_inference_backend,
            InferenceBackendType.TRITON: generate_triton_inference_backend,
        }

        backend_generator = type_map.get(self._type)
        if backend_generator is None:
            raise RuntimeError(
                f"{self._type} not a valid inference backend for Door Classifier"
            )
        return InferenceProvider(
            backend=backend_generator(),
            camera_uuid=camera_uuid,
            config=config,
            device=torch.device(self._device),
            training_config=json.loads(extra_files["model_config"]),
        )
