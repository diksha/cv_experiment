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

from lib.ml.inference.backends.torchscript import TorchscriptBackend
from lib.ml.inference.backends.triton import TritonInferenceBackend
from lib.ml.inference.factory.base import (
    InferenceBackendType,
    InferenceProviderFactoryBase,
)
from lib.ml.inference.tasks.ergonomic_carry_object.res_net34.inference_provider import (
    InferenceProvider,
)


class Resnet34InferenceProviderFactory(InferenceProviderFactoryBase):
    """Inference Provider Factory."""

    def get_inference_provider(
        self,
        model_path: str,
    ) -> InferenceProvider:
        """Get correct Resnet 34 inference provider class
        Args:
            model_path (str): path to model
        Returns:
            InferenceProvider: resnet 34 inference provider
        Raises:
            RuntimeError: type not valid inference provider type
        """

        def generate_torchscript_inference_backend() -> TorchscriptBackend:
            """Generates the torchscript inference backend for carry object

            Returns:
                TorchscriptBackend: the generated provider
            """
            return TorchscriptBackend(
                jit_file_path=model_path,
                device=torch.device(self._device),
                dtypes=[torch.float],
            )

        def generate_triton_inference_backend() -> TritonInferenceBackend:
            """Generates the triton inference backend for carry object

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
                f"{self._type} not a valid inference backend for carry object task"
            )
        return InferenceProvider(backend=backend_generator())
