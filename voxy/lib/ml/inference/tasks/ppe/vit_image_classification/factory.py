#
# Copyright 2020-2023 Voxel Labs, Inc.
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
from transformers import ViTFeatureExtractor

from lib.ml.inference.backends.torchscript import TorchscriptBackend
from lib.ml.inference.backends.triton import TritonInferenceBackend
from lib.ml.inference.factory.base import (
    InferenceBackendType,
    InferenceProviderFactoryBase,
)
from lib.ml.inference.tasks.ppe.vit_image_classification.inference_provider import (
    InferenceProvider,
)


class VITImageClassificationInferenceProviderFactory(
    InferenceProviderFactoryBase
):
    """Inference Provider Factory."""

    def get_inference_provider(
        self,
        model_path: str,
        feature_extractor_weights: str,
        preprocessing_padding: int,
    ) -> InferenceProvider:
        """Get correct fully connected unet inference provider class
        Args:
            model_path (str): path to model
            preprocessing_padding (int): amount to pad image crop during
                preprocessing
            feature_extractor_weights (str): name of the weights to initialize
                the ViTFeatureExtractor
        Returns:
            InferenceProvider: the vit image classification image provider
        Raises:
            RuntimeError: type not valid inference provider type
        """

        def generate_torchscript_inference_backend() -> TorchscriptBackend:
            """Generates the torchscript inference backend for spills

            Returns:
                TorchscriptBackend: the generated provider
            """
            return TorchscriptBackend(
                jit_file_path=model_path,
                device=torch.device(self._device),
                dtypes=[torch.float],
            )

        def generate_triton_inference_backend() -> TritonInferenceBackend:
            """Generates the triton inference backend for spills

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
                f"{self._type} not a valid inference backend for safety vest"
            )
        return InferenceProvider(
            backend=backend_generator(),
            feature_extractor=ViTFeatureExtractor.from_pretrained(
                feature_extractor_weights, return_tensors="pt"
            ),
            padding=preprocessing_padding,
            device=torch.device(self._device),
        )
