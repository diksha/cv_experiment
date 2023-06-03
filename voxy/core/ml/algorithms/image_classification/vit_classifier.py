#
# Copyright 2020-2022 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

import json
import typing
import uuid
from tempfile import mkdtemp

import pytorch_lightning as pl
import torch
from torch.optim.optimizer import Optimizer
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Snake Case Naming Style:
# trunk-ignore-all(pylint/C0103)

# Too many arguments:
# trunk-ignore-all(pylint/R0913)


class ViTImageClassifier(pl.LightningModule):
    """
    ViTImageClassifier. This is the huggingface transformer
    wrapper to be used in the model training framework
    """

    def __init__(
        self,
        num_labels: int,
        id_to_label: dict,
        label_to_id: dict,
        loss: str,
        optimizer: str,
        optimizer_params: dict,
    ):
        """
        Constructs ViT model

        Args:
            num_labels (int): the number of labels to be trained
            id_to_label (dict): conversion from id to label, needed to construct from pretrained
            label_to_id (dict): the conversion from label to id
            loss (typing.Type[torch.nn.Module]): the loss
            optimizer (typing.Type[Optimizer]): the optimizer to be used when training
            optimizer_params (dict): any extra kwargs to the optimizer other than parameters

        Raises:
           ValueError: if the loss is not found in torch.nn
           ValueError: if the optimizer is not found in the torch.optim
        """
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_labels,
            id2label=id_to_label,
            label2id=label_to_id,
            torchscript=True,
        )
        self.loss = getattr(torch.nn, loss)()
        if self.loss is None:
            raise ValueError(f"Loss {loss} was not found in torch.nn")
        self.optimizer = getattr(torch.optim, optimizer)
        if self.optimizer is None:
            raise ValueError(
                f"Optimizer {optimizer} was not found in torch.optim"
            )
        self.optimizer_kwargs = optimizer_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the network. This extracts the features from the input
        and passes it through the ViT model

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: the output tensor
        """
        return self.model(x)

    def training_step(
        self, batch: typing.Tuple[torch.Tensor], __batch_index: int
    ) -> torch.Tensor:
        """
        Training Step. This takes as input a batch and a batch index
        and returns the loss

        Args:
            batch (typing.Tuple[torch.Tensor]): current training batch

        Returns:
            torch.Tensor: the training loss for this batch
        """
        x, y = batch
        y_hat = self(x)[0]
        loss = self.loss(y_hat, y)
        self.log("training_loss", loss)
        return loss

    def validation_step(
        self, batch: typing.Tuple[torch.Tensor], __batch_index: int
    ) -> torch.Tensor:
        """
        Validation Step. This takes as input a batch and a batch index
        and returns the loss

        Args:
            batch (typing.Tuple[torch.Tensor]): current validation batch

        Returns:
            torch.Tensor: the validation loss for this batch
        """
        x, y = batch
        y_hat = self(x)[0]
        loss = self.loss(y_hat, y)
        self.log("validation_loss", loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        """
        Configures the optimizers for the run. This constructs the optimizer and
        passes in any input kwargs

        Returns:
            Optimizer: the optimizer required to train the model
        """
        return self.optimizer(self.parameters(), **self.optimizer_kwargs)

    def save(self, config: dict) -> str:
        """
        Saves the current model and config to disk

        Args:
            config (dict): the model config to save

        Returns:
            str: the local saved model path
        """
        # make a temp directory
        tempdir = mkdtemp(prefix="VIT_MODEL")
        model_path = f"{tempdir}/{str(uuid.uuid4())}.pth"
        _extra_files = {"model_config": json.dumps(config)}
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        random_tensor = torch.tensor(
            feature_extractor([torch.randn(3, 200, 200)])["pixel_values"][0]
        ).unsqueeze(0)
        traced_model = torch.jit.trace(self.model, random_tensor)
        traced_model.save(
            model_path,
            _extra_files=_extra_files,
        )
        return model_path

    @classmethod
    def load(cls, model_path: str) -> typing.Tuple[torch.nn.Module, dict]:
        """
        Loads the model from a local path

        Args:
            model_path (str): the local model path

        Returns:
            typing.Tuple[torch.nn.Module, dict]: the current model and its config
        """
        _extra_files = {"model_config": ""}
        model = torch.jit.load(model_path, _extra_files=_extra_files)

        return model, json.loads(_extra_files["model_config"])
