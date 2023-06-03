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

import copy
import json
import os
import random
import uuid
import warnings

import numpy as np
import torch
from loguru import logger
from sklearn.exceptions import UndefinedMetricWarning

# from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from core.ml.algorithms.image_classification.vanilla_resnet_50 import (
    VanillaResnet50Module,
)
from core.ml.data.loaders.common.registry import ConverterRegistry
from core.ml.experiments.tracking.experiment_tracking import ExperimentTracker
from core.ml.training.metrics.registry import get_conf_matrix, get_metrics
from core.ml.training.models.classifiermodels import ModelEMA
from core.ml.training.models.model import Model
from core.ml.training.models.model_training_result import ModelTrainingResult
from core.ml.training.registry.registry import ModelClassRegistry
from core.perception.inference.lib.vanilla_resnet_inference import (
    VanillaResnetInferenceModel,
)
from core.perception.inference.transforms.registry import get_transforms
from core.structs.dataset import DataloaderType
from core.structs.dataset import Dataset as VoxelDataset

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# It is high priority to remove a lot of this and refactor. Until then these will be ignored:
# trunk-ignore-all(pylint/R0902)
# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(pylint/W9012)
# trunk-ignore-all(pylint/C0103)
# trunk-ignore-all(semgrep/trailofbits.python.automatic-memory-pinning.automatic-memory-pinning)
# trunk-ignore-all(pylint/C0116)
# trunk-ignore-all(pylint/C0115)
# trunk-ignore-all(pylint/W0511)
# trunk-ignore-all(pylint/R0914)
# trunk-ignore-all(pylint/C0301)
# trunk-ignore-all(pylint/R0913)


@ModelClassRegistry.register()
class VanillaResnet50(Model):
    """
    Defines how the training of Vanilla Resnet happens.

    Note: Training works only with csv dataset and will fail currently for other datasets.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            config["device"] if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Device used is {self.device}")
        self.model = VanillaResnet50Module(
            num_classes=self.config["model_parameters"]["params"][
                "num_classes"
            ],
            freeze_layers=False,
        ).model
        self.model = self.model.to(self.device)
        self._setup_torch_runs()
        self.model_uuid = str(uuid.uuid4())
        os.makedirs(self.config["model_path"], exist_ok=True)
        self.model_path = "/".join(
            [
                self.config["model_path"],
                f"{self.model_uuid}.pt",
            ]
        )
        self.model_path_jit = "/".join(
            [
                self.config["model_path"],
                f"{self.model_uuid}-jit.pt",
            ]
        )
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.calculate_metrics = get_metrics(
            config["model_parameters"]["metrics"]
        )

    def _setup_torch_runs(self):
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        torch.cuda.manual_seed(self.config["seed"])
        torch.backends.cudnn.deterministic = True

    def _get_optimizer(self, model):
        return optim.Adam(
            params=[
                {
                    "params": model.conv1.parameters(),
                    "lr": self.config["optimizer"]["lr"] / 10,
                },
                {
                    "params": model.bn1.parameters(),
                    "lr": self.config["optimizer"]["lr"] / 10,
                },
                {
                    "params": model.layer1.parameters(),
                    "lr": self.config["optimizer"]["lr"] / 8,
                },
                {
                    "params": model.layer2.parameters(),
                    "lr": self.config["optimizer"]["lr"] / 6,
                },
                {
                    "params": model.layer3.parameters(),
                    "lr": self.config["optimizer"]["lr"] / 4,
                },
                {
                    "params": model.layer4.parameters(),
                    "lr": self.config["optimizer"]["lr"] / 2,
                },
                {"params": model.fc.parameters()},
            ],
            lr=self.config["optimizer"]["lr"],
        )

    def train(
        self, dataset: VoxelDataset, experiment_tracker: ExperimentTracker
    ) -> ModelTrainingResult:
        logger.info("Beginning Model Training")
        dataloader = ConverterRegistry.get_dataloader(
            dataset,
            self.dataloader_type(),
            transforms=get_transforms(self.config["data"]["transforms"]),
            augmentation_transforms=get_transforms(
                self.config["data"]["augmentation_transforms"]
            ),
        )
        data_loaders = {}
        data_loaders["train"] = dataloader.get_training_set()
        data_loaders["val"] = dataloader.get_validation_set()

        # TODO: make this less hardcoded
        num_epochs = self.config["optimizer"]["n_epoch"]

        optimizer = self._get_optimizer(self.model)
        best_valid_loss = float("inf")
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[p["lr"] for p in optimizer.param_groups],
            total_steps=num_epochs * len(data_loaders["train"]),
        )
        ema_model = ModelEMA(self.model)
        for epoch in tqdm(range(num_epochs)):
            experiment_tracker.log(
                message={"epoch": f"{epoch+1:02}"}, iteration=epoch + 1
            )
            train_results = self._train_epoch(
                self.model,
                ema_model,
                data_loaders["train"],
                scheduler=scheduler,
                optimizer=optimizer,
                labels=dataloader.get_label_ids(),
            )
            experiment_tracker.log(
                prefix="train", message=train_results, iteration=epoch + 1
            )
            val_results = self._evaluate_epoch(
                data_loaders["val"], labels=dataloader.get_label_ids()
            )
            experiment_tracker.log(
                message=val_results, prefix="val", iteration=epoch + 1
            )
            best_valid_loss = self._save_model(
                copy.deepcopy(self.model), val_results, best_valid_loss
            )
        logger.info("Finished Model Training")
        return ModelTrainingResult(saved_local_path=self.model_path_jit)

    @classmethod
    def dataloader_type(cls):
        return DataloaderType.PYTORCH_IMAGE

    def _save_model(self, model, val_results, best_valid_loss):
        # update model if validation loss is better
        if val_results["loss"] < best_valid_loss:
            best_valid_loss = val_results["loss"]
            torch.save(
                model.state_dict(),
                self.model_path,
            )
        best_model_wts = model.state_dict()
        model.load_state_dict(best_model_wts)
        example_input = torch.randn(1, 3, 224, 224, requires_grad=False).to(
            self.device
        )
        traced_model = torch.jit.trace(model, example_input)
        extra_files = {"model_config": json.dumps(self.config)}
        traced_model.save(self.model_path_jit, _extra_files=extra_files)
        return best_valid_loss

    def _evaluate_epoch(self, iterator, labels):
        """
        Runs the evaluation script for an epoch.

        Args:
            iterator (torch.utils.data.Dataloader): iterable dataloader to evaluate.

        Returns:
            tuple: (loss, acc_1, acc_2, f1)
        """

        epoch_loss_total = 0
        epoch_acc_1 = 0
        epoch_acc_2 = 0
        model_result = []
        targets = []
        self.model.eval()

        with torch.no_grad():
            for (x, y) in iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)

                loss = self.criterion(y_pred, y)
                epoch_loss_total += loss.item()

                result = self.calculate_metrics(
                    y_pred.cpu().numpy(), y.cpu().numpy(), labels=labels
                )

                epoch_acc_1 += result["top_1_accuracy"].item()
                epoch_acc_2 += result["top_k_accuracy"].item()
                model_result.extend(y_pred.cpu().numpy())
                targets.extend(y.cpu().numpy())

        results = self.calculate_metrics(
            np.array(model_result), np.array(targets), labels=labels
        )

        epoch_loss_total /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_2 /= len(iterator)

        return {
            "loss": epoch_loss_total,
            "acc_1": epoch_acc_1,
            "acc_2": epoch_acc_2,
            "f1": results["weighted/f1"],
            "precision": results["weighted/precision"],
            "recall": results["weighted/recall"],
        }

    def _train_epoch(
        self, model, ema_model, iterator, scheduler, optimizer, labels
    ):
        epoch_loss_total = 0
        epoch_acc_1 = 0
        epoch_acc_2 = 0
        model_result = []
        targets = []
        model.train()
        for (x, y) in iterator:

            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            y_pred = model(x)

            loss = self.criterion(y_pred, y)
            loss.backward()

            result = self.calculate_metrics(
                y_pred.cpu().detach().numpy(), y.cpu().numpy(), labels
            )

            ema_model.update(model)

            optimizer.step()

            scheduler.step()

            # todo: move to torch.no_grad block
            epoch_loss_total += loss.item()
            epoch_acc_1 += result["top_1_accuracy"].item()
            epoch_acc_2 += result["top_k_accuracy"].item()
            model_result.extend(y_pred.cpu().detach().numpy())
            targets.extend(y.cpu().numpy())

        epoch_loss_total /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_2 /= len(iterator)
        results = self.calculate_metrics(
            np.array(model_result), np.array(targets), labels=labels
        )
        return {
            "loss": epoch_loss_total,
            "acc_1": epoch_acc_1,
            "acc_2": epoch_acc_2,
            "f1": results["weighted/f1"],
            "precision": results["weighted/precision"],
            "recall": results["weighted/recall"],
        }

    @classmethod
    def evaluate(
        cls,
        model_training_result: ModelTrainingResult,
        dataset: VoxelDataset,
        experiment_tracker: ExperimentTracker,
    ) -> dict:
        """
        Evaluates the model based on the result of training
        the dataset and tracks it using the experiment tracker

        Args:
            model_training_result (ModelTrainingResult): the result of training (model path, config, etc.)
            dataset (VoxelDataset): the dataset used to evaluate
            experiment_tracker (ExperimentTracker): the experiment tracker to track metrics

        Returns:
            dict: a dictionary of metrics
        """
        # this should be cached if we had already downloaded during training
        dataloader = ConverterRegistry.get_dataloader(
            dataset, cls.dataloader_type()
        )
        data_loaders = {}
        data_loaders["test"] = dataloader.get_test_set()

        model_results = []
        targets = []

        _extra_files = {"model_config": ""}
        model = torch.jit.load(
            model_training_result.saved_local_path, _extra_files=_extra_files
        )
        model_config = json.loads(_extra_files["model_config"])

        inference_model = VanillaResnetInferenceModel(model, model_config)
        for (x, y) in dataloader.get_image_label_iterator(
            data_loaders["test"].dataset
        ):
            predictions = inference_model.infer(x)
            model_results.append(predictions.detach().numpy())
            targets.extend([y])
        calculate_metrics = get_metrics(
            model_config["model_parameters"]["metrics"]
        )
        results = calculate_metrics(
            np.array(model_results),
            np.array(targets),
            labels=dataloader.get_label_ids(),
        )
        experiment_tracker.log_table(
            message=results,
        )
        if model_config["model_parameters"].get("conf_matrix", False):
            classes = ["Closed", "Open", "Partially Open"]  # Sorted classes
            calulate_conf_matrix = get_conf_matrix()
            conf_matrix = calulate_conf_matrix(
                np.array(model_results), np.array(targets)
            )
            experiment_tracker.log_confusion_matrix(
                conf_matrix,
                model_config["model_parameters"]["type"],
                "Predictions",
                "Ground Truths",
                classes,
            )
        return results
