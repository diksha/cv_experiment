#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import typing
from dataclasses import dataclass

import numpy as np
import torch
import tritonclient
from tritonclient.grpc import (
    InferInput,
    InferRequestedOutput,
    model_config_pb2,
)
from tritonclient.grpc.model_config_pb2 import ModelInput, ModelOutput

from core.structs.model import ModelConfiguration
from lib.infra.utils.resolve_model_path import unresolve_model_path
from lib.ml.inference.backends.base import InferenceBackend
from lib.utils.triton.triton_model_name import triton_model_name

# For more information on triton protobufs please see:
# https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto


# Triton only supports certain file names when the file itself is sent over
# GRPC, it will fail with a file not found error since the temp file it creates
# are named these depending on the platform name
PLATFORM_TARGET_NAME_MAP = {
    "pytorch_libtorch": "model.pt",
    "tensorrt_plan": "model.plan",
}
# NOTE: torch doesn't support unsigned integer types
#       so uint16-64 are signed integer types
TRITON_DTYPE_TO_TORCH_DTYPE = {
    model_config_pb2.DataType.TYPE_INVALID: None,
    model_config_pb2.DataType.TYPE_BOOL: torch.bool,
    model_config_pb2.DataType.TYPE_UINT8: torch.uint8,
    model_config_pb2.DataType.TYPE_UINT16: torch.int16,
    model_config_pb2.DataType.TYPE_UINT32: torch.int32,
    model_config_pb2.DataType.TYPE_UINT64: torch.int64,
    model_config_pb2.DataType.TYPE_INT8: torch.int8,
    model_config_pb2.DataType.TYPE_INT16: torch.int16,
    model_config_pb2.DataType.TYPE_INT32: torch.int32,
    model_config_pb2.DataType.TYPE_INT64: torch.int64,
    model_config_pb2.DataType.TYPE_FP16: torch.float16,
    model_config_pb2.DataType.TYPE_FP32: torch.float32,
    model_config_pb2.DataType.TYPE_FP64: torch.float64,
    model_config_pb2.DataType.TYPE_STRING: torch.uint8,
    model_config_pb2.DataType.TYPE_BF16: torch.bfloat16,
}

TRITON_DTYPE_TO_NUMPY_DTYPE = {
    model_config_pb2.DataType.TYPE_BOOL: np.bool,
    model_config_pb2.DataType.TYPE_UINT8: np.uint8,
    model_config_pb2.DataType.TYPE_UINT16: np.uint16,
    model_config_pb2.DataType.TYPE_UINT32: np.uint32,
    model_config_pb2.DataType.TYPE_UINT64: np.uint64,
    model_config_pb2.DataType.TYPE_INT8: np.int8,
    model_config_pb2.DataType.TYPE_INT16: np.int16,
    model_config_pb2.DataType.TYPE_INT32: np.int32,
    model_config_pb2.DataType.TYPE_INT64: np.int64,
    model_config_pb2.DataType.TYPE_FP16: np.float16,
    model_config_pb2.DataType.TYPE_FP32: np.float32,
    model_config_pb2.DataType.TYPE_FP64: np.float64,
    model_config_pb2.DataType.TYPE_STRING: np.uint8,
    model_config_pb2.DataType.TYPE_BF16: np.float32,
}


@dataclass
class TritonInputType:
    """
    Intermediate type for triton model inputs and triton infer inputs. For more information on
    triton protobufs please see:

    https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
    """

    input_name: str
    input_size: typing.List[int]
    input_format: str
    input_dtype_numpy: np.dtype

    @classmethod
    def from_triton_config(cls, triton_input: ModelInput) -> "TritonInputType":
        """
        Generates the triton config from the triton input

        Args:
            triton_input (ModelInput): the triton input

        Returns:
            TritonInputType: the triton input type
        """
        return cls(
            input_name=triton_input.name,
            input_size=list(triton_input.dims),
            input_format=model_config_pb2.DataType.Name(
                triton_input.data_type
            ).lstrip("TYPE_"),
            input_dtype_numpy=TRITON_DTYPE_TO_NUMPY_DTYPE[
                triton_input.data_type
            ],
        )

    @classmethod
    def to_infer_inputs(
        cls,
        triton_inputs: typing.List["TritonInputType"],
        triton_input_tensors: typing.List[torch.Tensor],
    ) -> typing.List[InferInput]:
        """
        Generates the inference inputs

        Args:
            triton_inputs (typing.List[TritonInputType]): the list of triton inputs
            triton_input_tensors (typing.List[torch.Tensor]): the list of torch tensors

        Raises:
           ValueError: if the number of triton inputs and triton input tensors are not the same

        Returns:
            typing.List[InferInput]: the list of inference inputs for triton
        """
        if len(triton_input_tensors) != len(triton_inputs):
            raise ValueError(
                "There are not corresponding inputs for all triton inputs"
            )
        input_types = []
        for triton_input_tensor, triton_input in zip(
            triton_input_tensors, triton_inputs
        ):
            input_type = InferInput(
                triton_input.input_name,
                triton_input_tensor.size(),
                triton_input.input_format,
            )
            input_type.set_data_from_numpy(
                triton_input_tensor.numpy().astype(
                    dtype=triton_input.input_dtype_numpy
                )
            )
            input_types.append(input_type)
        return input_types


@dataclass
class TritonOutputType:
    """
    Intermediate type for triton model outputs and triton infer outputs. For more information on
    triton protobufs please see:

    https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
    """

    output_name: str

    @classmethod
    def from_triton_config(
        cls, triton_output: ModelOutput
    ) -> "TritonOutputType":
        """
        Generates this class from the
        ModelOutput fron triton config

        Args:
            triton_output (ModelOutput): the triton output

        Returns:
            TritonOutputType: the triton output type
        """
        return cls(
            output_name=triton_output.name,
        )

    @classmethod
    def to_requested_outputs(
        cls, triton_outputs: typing.List["TritonOutputType"]
    ) -> typing.List[InferRequestedOutput]:
        """
        Generates the requested outputs for triton

        Args:
            triton_outputs (typing.List[TritonOutputType]): the outputs grabbed from the config

        Returns:
            typing.List[InferRequestedOutput]: the inference outputs
        """
        return [
            InferRequestedOutput(triton_output.output_name)
            for triton_output in triton_outputs
        ]


class TritonInferenceBackend(InferenceBackend):
    """
    Basic wrapper interface for a backend in triton
    """

    def __init__(
        self,
        model_name: str,
        triton_server_url: str,
        is_ensemble: bool = False,
    ):
        """
        Initializes triton backend model

        Args:
            model_name (str): The model name
                        (
                        e.g.: artifacts_02_27_2023_michaels_wesco_office_yolo
                         /best_736_1280.engine
                         )
            is_ensemble (bool): Whether the model is an ensemble

        Raises:
           RuntimeError: As the class is not implemented
        """
        self.triton_client = tritonclient.grpc.InferenceServerClient(
            url=triton_server_url, verbose=False, ssl=False
        )
        # try to serve the model
        self.model_name = triton_model_name(
            unresolve_model_path(model_name), is_ensemble
        )
        self.triton_client.get_model_metadata(
            model_name=self.model_name, model_version="1"
        )
        triton_config = self.triton_client.get_model_config(
            model_name=self.model_name, model_version="1"
        ).config
        self.input_types = [
            TritonInputType.from_triton_config(triton_input)
            for triton_input in triton_config.input
        ]

        self.output_types = [
            TritonOutputType.from_triton_config(triton_output)
            for triton_output in triton_config.output
        ]

    def get_config(self) -> ModelConfiguration:
        """
        Loads a model configuration relating to all relevant preprocessing
        postprocessing for the model

        This is related to how to configure preprocessing/post processing
        since that may change from model to model
        (different transforms, etc.)

        Raises:
           RuntimeError: As the class is not implemented
        """
        # TODO: update this to grab the proper model configuration
        raise RuntimeError("Not Implemented!")

    def infer(
        self, input_tensors: typing.List[torch.Tensor]
    ) -> typing.List[torch.Tensor]:
        """
        Performs inference on the input tensor x and generates the model
        result

        Args:
            input_tensors (typing.List[torch.Tensor]): The input tensor to perform inference on.

        Raises:
           RuntimeError: As the class is not implemented

        Returns:
           typing.List[torch.Tensor]: the inference result of the model that was run
                    remotely on the triton server
        """
        inputs = TritonInputType.to_infer_inputs(
            self.input_types, input_tensors
        )
        outputs = TritonOutputType.to_requested_outputs(self.output_types)
        results = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs,
        )
        # the raw output leads to undefined behavior since the numpy reference is not writable
        # so we have to copy this
        return [
            torch.from_numpy(
                np.array(results.as_numpy(output_type.output_name), copy=True)
            )
            for output_type in self.output_types
        ]
