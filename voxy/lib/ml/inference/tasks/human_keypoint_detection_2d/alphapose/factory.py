import torch

from lib.ml.inference.backends.torchscript import TorchscriptBackend
from lib.ml.inference.backends.triton import TritonInferenceBackend
from lib.ml.inference.factory.base import (
    InferenceBackendType,
    InferenceProviderFactoryBase,
)
from lib.ml.inference.tasks.human_keypoint_detection_2d.alphapose.inference_provider import (
    InferenceProvider,
)


class AlphaposeInferenceProviderFactory(InferenceProviderFactoryBase):
    """Inference Provider Factory."""

    def get_inference_provider(
        self,
        model_path: str,
    ) -> InferenceProvider:
        """Get correct yolo inference provider class
        Args:
            model_path (str): path to model
        Returns:
            typing.Any: Alphapose inference provider
        Raises:
            RuntimeError: type not valid inference provider type
        """

        def generate_torchscript_inference_backend() -> TorchscriptBackend:
            """Generates the torchscript backend for human keypoint detection 2d

            Returns:
                TorchscriptBackend: the generated provider
            """
            return TorchscriptBackend(
                jit_file_path=model_path,
                device=torch.device(self._device),
                dtypes=[torch.float],
            )

        def generate_triton_inference_backend() -> TritonInferenceBackend:
            """Generates the triton backend for human keypoint detection 2d

            Returns:
                TritonInferenceBackend: the generated provider
            """
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
                f"{self._type} not a valid inference backend for Alphapose"
            )
        return InferenceProvider(backend=backend_generator())
