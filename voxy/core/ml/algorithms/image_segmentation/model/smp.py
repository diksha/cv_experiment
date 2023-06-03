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

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from loguru import logger
from pytorch_lightning.trainer.supporters import CombinedLoader

from core.utils.aws_utils import upload_directory_to_s3


class SmpModel(pl.LightningModule):
    def __init__(
        self,
        arch="Unet",
        encoder_name="resnet18",
        in_channels=3,
        out_classes=1,
        data_loaders=None,
        model_checkpoints_s3_relative=None,
        model_local_checkpoints_dir=None,
        **kwargs,
    ):
        """Create the smp model using pytorch lightning
        Args:
            arch (str): Segmentation architecture. Default Unet
            encoder_name (str): Encoder model used. Default resnet18
            in_channels (int): Number of channels in input image (3 for RGB)
            out_classes (int): Number of classes - 1
            data_loaders (_type_, optional): list of dataloaders. Defaults to None.
        """
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer(
            "std", torch.tensor(params["std"]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "mean", torch.tensor(params["mean"]).view(1, 3, 1, 1)
        )

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.BINARY_MODE, from_logits=True
        )
        self.dataloaders_train_dict = {
            str(i): data_loaders[i]["train"] for i in range(len(data_loaders))
        }
        self.dataloaders_val_dict = {
            str(i): data_loaders[i]["val"] for i in range(len(data_loaders))
        }
        self.model_checkpoints_s3_relative = model_checkpoints_s3_relative
        self.local_checkpoints_dir = model_local_checkpoints_dir

    def train_dataloader(self):
        """Setup Training Dataloaders

        Returns:
            _type_: combined dataloader of training
        """
        return CombinedLoader(
            self.dataloaders_train_dict, mode="max_size_cycle"
        )

    def val_dataloader(self):
        """Setup validation Dataloaders

        Returns:
            _type_: combined dataloader of validation
        """
        return CombinedLoader(self.dataloaders_val_dict, mode="max_size_cycle")

    def forward(self, image):
        """A forward pass of the model

        Args:
            image (torch.tensor): Input image to model

        Returns:
            torch.tensor: Model prediction on input image
        """
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        """Shared training and validation step

        Args:
            batch (torch.tensor): Batch of input images and ground truth annotations
            stage (str): Indicates a training or validation stage

        Returns:
            dict: model metric for step
        """
        loss = []
        true_positive_all = torch.tensor([]).cuda()
        false_positve_all = torch.tensor([]).cuda()
        false_negative_all = torch.tensor([]).cuda()
        true_negative_all = torch.tensor([]).cuda()
        for batch_dl_idx in batch:
            image = batch[batch_dl_idx][0]
            mask = batch[batch_dl_idx][1]
            logits_mask = self.forward(image)
            loss.append(self.loss_fn(logits_mask, mask))
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()
            (
                true_positive,
                false_positve,
                false_negative,
                true_negative,
            ) = smp.metrics.get_stats(
                pred_mask.long(), mask.long(), mode="binary"
            )
            true_positive_all = torch.cat((true_positive_all, true_positive))
            false_positve_all = torch.cat((false_positve_all, false_positve))
            false_negative_all = torch.cat(
                (false_negative_all, false_negative)
            )
            true_negative_all = torch.cat((true_negative_all, true_negative))
        loss = sum(loss) / len(loss)
        return {
            "loss": loss,
            "true_positive": true_negative_all,
            "false_positve": false_positve_all,
            "false_negative": false_negative_all,
            "true_negative": true_negative_all,
        }

    def shared_epoch_end(self, outputs, stage):
        """Aggregates the metrics for a particular step

        Args:
            outputs (list): List of model metrics through the epoch
            stage (str): Indicates training or validation stage
        """
        true_positive = torch.cat([x["true_positive"] for x in outputs])
        false_positve = torch.cat([x["false_positve"] for x in outputs])
        false_negative = torch.cat([x["false_negative"] for x in outputs])
        true_negative = torch.cat([x["true_negative"] for x in outputs])
        per_image_iou = smp.metrics.iou_score(
            true_positive,
            false_positve,
            false_negative,
            true_negative,
            reduction="micro-imagewise",
        )
        dataset_iou = smp.metrics.iou_score(
            true_positive,
            false_positve,
            false_negative,
            true_negative,
            reduction="micro",
        )

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        self.log(f"{stage}_metrics", metrics)

    def training_step(self, batch, batch_idx):
        """Evaluates single training step metrics

        Args:
            batch (torch.tensor): A batch containing the image and annotation
            batch_idx (int): The batch index

        Returns:
            dict: returns metrics on the batch
        """
        metrics = self.shared_step(batch, "train")
        self.log("train_loss", metrics["loss"])
        return metrics

    def training_epoch_end(self, outputs):
        """Aggregates training metrics from entire epoch

        Args:
            outputs (torch.tensor): Model output results

        Returns:
            dict: returns metrics on the batch
        """
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        """Evaluates single validation step metrics

        Args:
            batch (torch.tensor): A batch containing the image and annotation
            batch_idx (int): The batch index

        Returns:
            dict: returns metrics on the batch
        """
        metrics = self.shared_step(batch, "valid")
        self.log("valid_loss", metrics["loss"])
        return metrics

    def validation_epoch_end(self, outputs):
        """Validation epoch end

        Args:
            outputs (torch.tensor): Model output results
        Returns:
            dict: returns metrics on the batch
        """
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        """Evaluates single test step metrics

        Args:
            batch (torch.tensor): A batch containing the image and annotation
            batch_idx (int): The batch index

        Returns:
            dict: returns metrics on the batch
        """
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        """Aggregates test metrics from entire epoch

        Args:
            outputs (torch.tensor): Model output results

        Returns:
            dict: model metrics from entire test epoch
        """
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """Set up training optimizer

        Returns:
            torch.optim.Adam: A configured adam optimizer with parameters
        """
        return torch.optim.Adam(self.parameters(), lr=0.0001)

    def on_epoch_end(self):
        """Saving Model Checkpoints on S3"""
        logger.info(
            f"Syncing Checkpoints to {self.model_checkpoints_s3_relative}"
        )
        upload_directory_to_s3(
            "voxel-users",
            self.local_checkpoints_dir,
            self.model_checkpoints_s3_relative,
        )
