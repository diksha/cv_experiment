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

import argparse
import copy
import os
import random

import numpy as np
import torch
import wandb
import yaml
from loguru import logger
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from core.ml.data.loaders.resources.dataloaders.csv_dataloader import (
    ImagefromCSV,
)
from core.ml.training.metrics.registry import get_metrics
from core.ml.training.models.classifiermodels import ModelEMA
from core.ml.training.models.registry import get_model_deprecated
from core.perception.inference.transforms.registry import get_transforms
from core.utils.logging.slack.get_slack_webhooks import (
    get_perception_verbose_sync_webhook,
)
from core.utils.logging.slack.synchronous_webhook_wrapper import (
    SynchronousWebhookWrapper,
)


def get_dataloader(__config: str) -> ImagefromCSV:
    """
    Returns the image from csv dataloader for
    backwards compatibility

    Returns:
        ImagefromCSV: the image from csv class
    """
    return ImagefromCSV


class TrainClassifier:
    def __init__(self, config):
        """Initializes the training class.

        Args:
            config (str): a YAML door training config file.
        """
        self.config = config
        self.device = torch.device(
            config["device"] if torch.cuda.is_available() else "cpu"
        )

        wandb.init(**config["wandb"]["init_params"])

        # Make runs reproducible
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        torch.backends.cudnn.deterministic = True

        # Model
        _model = get_model_deprecated(config["model"]["type"])
        Classifier = _model(
            num_classes=config["model"]["params"]["num_classes"],
            freeze_layers=False,
        )
        model = Classifier.model

        # Optimizer and Loss Criterion
        if config["mode"] == "train":
            params = [
                {
                    "params": model.conv1.parameters(),
                    "lr": config["optimizer"]["lr"] / 10,
                },
                {
                    "params": model.bn1.parameters(),
                    "lr": config["optimizer"]["lr"] / 10,
                },
                {
                    "params": model.layer1.parameters(),
                    "lr": config["optimizer"]["lr"] / 8,
                },
                {
                    "params": model.layer2.parameters(),
                    "lr": config["optimizer"]["lr"] / 6,
                },
                {
                    "params": model.layer3.parameters(),
                    "lr": config["optimizer"]["lr"] / 4,
                },
                {
                    "params": model.layer4.parameters(),
                    "lr": config["optimizer"]["lr"] / 2,
                },
                {"params": model.fc.parameters()},
            ]
        self.optimizer = (
            optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])
            if config["mode"] == "lr_find"
            else optim.Adam(params, lr=config["optimizer"]["lr"])
        )
        self.criterion = (
            nn.CrossEntropyLoss(label_smoothing=0.3)
            if config["loss"]["label_smooth"] is True
            else nn.CrossEntropyLoss()
        )

        self.model = model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.best_valid_loss = float("inf")
        # TODO: Add to config acc_f1_prec_rec
        self.calculate_metrics = get_metrics(config["model"]["metrics"])

    def _log_to_wandb_and_console(
        self, loss, acc_1, acc_2, f1, precision, recall, prefix=""
    ):
        """Logs the metrics to wandb and console

        Args:
            loss (float): loss of the split.
            acc_1 (float): accuracy of the split.
            acc_2 (float): accuracy of the split.
            f1 (float): f1 score of the split.
            precision (float): precision of the split.
            recall (float): recall of the split.
            prefix (str): Prefix for the metrics split. Defaults to "".
        """
        # Log to WandB
        wandb.log({prefix + " loss": loss})
        wandb.log({prefix + " f1": f1})
        wandb.log({prefix + " precision": f1})
        wandb.log({prefix + " recall": f1})
        wandb.log({prefix + " acc @1": acc_1})
        wandb.log({prefix + " acc @2": acc_2})

        # Log to console
        logger.info(
            f"\t {prefix} Loss: {loss:.3f} | {prefix} F1: {f1:.3f} | {prefix} Precision: {precision:.3f} |{prefix} Recall: {recall:.3f} | {prefix} Acc @1: {acc_1*100:6.2f}% | "
            f"{prefix} Acc @2: {acc_2*100:6.2f}%"
        )

    def _log_confusion_to_wandb(self, y, y_pred):
        """Logs the confusion matrix to wandb."""
        wandb.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=np.array(y),
                    preds=np.argmax(y_pred, 1),
                    class_names=["closed", "open", "partially_open"],
                )
            }
        )

    def notify_test_metrics(self, f1, precision, recall):
        """Notifies the test metrics to Slack.

        Args:
            f1 (float): f1 score of the split.
            precision (float): precision of the split.
            recall (float): recall of the split.
        """

        slack_webhook_notifier = SynchronousWebhookWrapper(
            get_perception_verbose_sync_webhook()
        )
        notification_block = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Door Classifier Performance, Model: {self.config['model']['name']}",
                },
            }
        ]

        # Add metrics to notification block
        metrics_fields = []
        metrics_fields.append(
            {
                "type": "mrkdwn",
                "text": f"WandB Link: <{wandb.run.get_url()}|{wandb.run.get_url()}>",
            }
        )
        metrics_fields.append(
            {
                "type": "mrkdwn",
                "text": f"F1: {f1:.3f}",
            },
        )
        metrics_fields.append(
            {
                "type": "mrkdwn",
                "text": f"Precision: {precision:.3f}",
            },
        )
        metrics_fields.append(
            {
                "type": "mrkdwn",
                "text": f"Recall: {recall:.3f}",
            },
        )

        notification_block.append(
            {
                "type": "section",
                "fields": metrics_fields,
            }
        )

        # Publish notification
        slack_webhook_notifier.post_message_block(notification_block)

    def _save_traced_model(self, model):
        """Saves the traced model to a file.

        Args:
            model (torch.nn.Module): Pytorch model to save.
        """
        best_model_wts = model.state_dict()
        self.model.load_state_dict(best_model_wts)
        example_input = torch.randn(1, 3, 224, 224, requires_grad=False).to(
            self.device
        )
        traced_model = torch.jit.trace(self.model, example_input)
        traced_model.save(
            f'{self.config["model_path"]}/{self.config["model"]["name"]}-jit.pth'
        )

    def test(self):
        """Tests the model."""
        data_loaders, _ = self._create_dataloaders(True)
        # load model
        self.model.load_state_dict(
            torch.load(
                f'{self.config["model_path"]}/{self.config["model"]["name"]}.pt'
            )
        )

        # Do inference on test split
        (
            model_result,
            targets,
            _,
            _,
            _,
        ) = self._do_inference_split(iterator=data_loaders["test"])

        # Get metrics
        (
            test_loss,
            test_acc_1,
            test_acc_2,
            test_f1,
            test_precision,
            test_recall,
        ) = self._evaluate_epoch(iterator=data_loaders["test"])

        self._log_to_wandb_and_console(
            test_loss,
            test_acc_1,
            test_acc_2,
            test_f1,
            test_precision,
            test_recall,
            prefix="test",
        )

        self._log_confusion_to_wandb(targets, model_result)

        self.notify_test_metrics(test_f1, test_precision, test_recall)

    def train(self):
        """Runs the training for n_epochs and saves the best model."""
        data_loaders, dataset_sizes = self._create_dataloaders()
        num_epochs = self.config["optimizer"]["n_epoch"]

        # scheduling
        scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[p["lr"] for p in self.optimizer.param_groups],
            total_steps=num_epochs * dataset_sizes["train"],
        )
        # TODO : Clean up model ema registry
        ema_model = ModelEMA(self.model)
        with tqdm(total=num_epochs) as progress_bar:
            for epoch in range(num_epochs):
                logger.info(f"Epoch: {epoch+1:02}")

                (
                    train_loss,
                    train_acc_1,
                    train_acc_2,
                    train_f1,
                    train_precision,
                    train_recall,
                ) = self._train_epoch(
                    ema_model,
                    iterator=data_loaders["train"],
                    scheduler=scheduler,
                )

                (
                    valid_loss,
                    valid_acc_1,
                    valid_acc_2,
                    valid_f1,
                    valid_precision,
                    valid_recall,
                ) = self._evaluate_epoch(iterator=data_loaders["val"])

                # Update model if validation loss is better
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    torch.save(
                        self.model.state_dict(),
                        f'{self.config["model_path"]}/{self.config["model"]["name"]}.pt',
                    )

                self._save_traced_model(copy.deepcopy(self.model))

                # Log Train & Validation Results
                self._log_to_wandb_and_console(
                    train_loss,
                    train_acc_1,
                    train_acc_2,
                    train_f1,
                    train_precision,
                    train_recall,
                    prefix="train",
                )
                self._log_to_wandb_and_console(
                    valid_loss,
                    valid_acc_1,
                    valid_acc_2,
                    valid_f1,
                    valid_precision,
                    valid_recall,
                    prefix="val",
                )

                progress_bar.update(1)

    def _do_inference_split(self, iterator):
        """Runs inference on the given split.

        Args:
            iterator (torch.utils.data.DataLoader): iterable dataloader to evaluate.

        Returns:
            tuple: (model_result, targets, epoch_loss_total, epoch_acc_1, epoch_acc_2)
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

                result = self.calculate_metrics(
                    y_pred.cpu().numpy(), y.cpu().numpy(), labels=[0, 1, 2]
                )

                epoch_loss_total += loss.item()
                epoch_acc_1 += result["top_1_accuracy"].item()
                epoch_acc_2 += result["top_k_accuracy"].item()
                model_result.extend(y_pred.cpu().numpy())
                targets.extend(y.cpu().numpy())

        epoch_loss_total /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_2 /= len(iterator)

        return (
            model_result,
            targets,
            epoch_loss_total,
            epoch_acc_1,
            epoch_acc_2,
        )

    def _evaluate_epoch(self, iterator):
        """Runs the evaluation script for an epoch.

        Args:
            iterator (torch.utils.data.DataLoader): iterable dataloader to evaluate.

        Returns:
            tuple: (loss, acc_1, acc_2, f1)
        """

        (
            model_result,
            targets,
            epoch_loss_total,
            epoch_acc_1,
            epoch_acc_2,
        ) = self._do_inference_split(iterator)

        results = self.calculate_metrics(
            np.array(model_result), np.array(targets), labels=[0, 1, 2]
        )

        return (
            epoch_loss_total,
            epoch_acc_1,
            epoch_acc_2,
            results["weighted/f1"],
            results["weighted/precision"],
            results["weighted/recall"],
        )

    def _train_epoch(self, ema_model, iterator, scheduler):
        """Runs the training step for asingle epoch.

        Args:
            ema_model (core.ml.training.models.classifiermodels.ModelEMA): _description_
            iterator (torch.utils.data.DataLoader): iterable dataloader to train.
            scheduler (lr_scheduler.OneCycleLR): Learning rate scheduler.

        Returns:
            _type_: _description_
        """
        epoch_loss_total = 0
        epoch_acc_1 = 0
        epoch_acc_2 = 0
        model_result = []
        targets = []
        self.model.train()
        for (x, y) in iterator:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)

            result = self.calculate_metrics(
                y_pred.cpu().detach().numpy(),
                y.cpu().numpy(),
                labels=[0, 1, 2],
            )

            loss.backward()
            ema_model.update(self.model)

            self.optimizer.step()

            scheduler.step()

            # TODO: move to torch.no_grad block
            epoch_loss_total += loss.item()
            epoch_acc_1 += result["top_1_accuracy"].item()
            epoch_acc_2 += result["top_k_accuracy"].item()
            model_result.extend(y_pred.cpu().detach().numpy())
            targets.extend(y.cpu().numpy())

        epoch_loss_total /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_2 /= len(iterator)
        results = self.calculate_metrics(
            np.array(model_result), np.array(targets), labels=[0, 1, 2]
        )

        return (
            epoch_loss_total,
            epoch_acc_1,
            epoch_acc_2,
            results["weighted/f1"],
            results["weighted/precision"],
            results["weighted/recall"],
        )

    def _calculate_norms(self):
        """Calculates the norms of the dataset

        Returns:
            tuple: mean and standard deviation of the dataset.
        """
        data_dir = self.config["data"]["dataloader_params"]["data_path"]
        dataset_loader = get_dataloader(self.config["data"]["dataloader_name"])
        train_data = dataset_loader(
            csv_path=data_dir,
            images_folder=data_dir,
            transform=transforms.ToTensor(),
            split="train",
        )

        means = torch.zeros(3)
        stds = torch.zeros(3)
        for img, _ in train_data:
            means += torch.mean(img, dim=(1, 2))
            stds += torch.std(img, dim=(1, 2))

        means /= len(train_data)
        stds /= len(train_data)

        return means.tolist(), stds.tolist()

    def _random_split(self, params, image_data):
        """Randomly splits the dataset into training and validation.

        Args:
            params (dict): data configuration parameters.
            image_data (dict): dictionary of image data.

        Returns:
            dict: dictionary of image data split into training and validation.
        """

        train_data = image_data["train"]
        n_train_examples = int(len(train_data) * (1 - params["valid_pct"]))
        n_valid_examples = len(train_data) - n_train_examples

        image_data["train"], image_data["val"] = data.random_split(
            train_data, [n_train_examples, n_valid_examples]
        )
        return image_data

    def _create_dataloaders(self, is_test=False):
        """Creates the dataloaders for the training and validation.

        Returns:
            dict: dictionary of training and validation dataloaders.
        """
        data_config = copy.deepcopy(self.config["data"])

        # Add Normalization to Transforms
        if data_config["normalize"]:
            mean, std = (
                data_config["normalize"]["mean"],
                data_config["normalize"]["std"],
            )
        else:
            mean, std = self._calculate_norms()

        data_config["train_transform"]["item_tfms"].append(
            {"name": "Normalize", "params": {"mean": mean, "std": std}}
        )

        data_config["test_transform"]["item_tfms"].append(
            {"name": "Normalize", "params": {"mean": mean, "std": std}}
        )

        # Get data transforms
        data_transforms = {}
        data_transforms["train"] = data_config.pop("train_transform")
        data_transforms["train"] = get_transforms(
            data_transforms["train"].pop("item_tfms", [])
        )
        data_transforms["test"] = data_config.pop("test_transform")
        data_transforms["test"] = get_transforms(
            data_transforms["test"].pop("item_tfms", [])
        )

        # Load in all train data, to be split into train/val
        name = data_config.pop("dataloader_name")
        dataset_class = get_dataloader(name)
        image_data = {}

        image_data["train"] = dataset_class(
            csv_path=self.config["data"]["dataloader_params"]["data_path"],
            images_folder=self.config["data"]["dataloader_params"][
                "data_path"
            ],
            split="train",
            transforms=data_transforms["train"],
        )

        # Load test data
        image_data["test"] = dataset_class(
            csv_path=self.config["data"]["dataloader_params"]["data_path"],
            images_folder=self.config["data"]["dataloader_params"][
                "data_path"
            ],
            split="test",
            use_weighted_random_sampler=self.config["data"][
                "dataloader_params"
            ]["use_weighted_random_sampler"],
            transforms=data_transforms["test"],
        )

        # split the train into valid/train
        params = data_config.pop("dataloader_params")
        dataloaders = {}
        dataset_sizes = {}
        if not is_test:
            image_data = self._random_split(params, image_data)
            weighted_samplers = {}

            # Create train & val dataloaders
            for phase in ["train", "val"]:

                # Create weighted samplers
                shuffle = False
                target = np.array(image_data[phase])[:, 1]

                class_sample_count = np.array(
                    [len(np.where(target == t)[0]) for t in np.unique(target)]
                )
                weight = 1.0 / class_sample_count

                samples_weight = np.array([weight[t] for t in target])

                samples_weight = torch.from_numpy(samples_weight).double()
                weighted_sampler = WeightedRandomSampler(
                    samples_weight, len(samples_weight)
                )
                weighted_samplers[phase] = weighted_sampler
                weighted_samplers[phase] = None

                # Create dataloaders
                dataloaders[phase] = data.DataLoader(
                    image_data[phase],
                    shuffle=shuffle,
                    batch_size=params["batch_size"],
                    sampler=weighted_samplers[phase],
                )

                dataset_sizes[phase] = len(image_data[phase])

        # Create test dataloaders
        dataloaders["test"] = data.DataLoader(
            image_data["test"],
        )
        dataset_sizes["test"] = len(image_data["test"])

        return dataloaders, dataset_sizes


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.ArgumentParser: Parsed arguments struct.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--camera_uuid", "-ca", type=str, required=True)
    parser.add_argument("--data_path", "-i", type=str, required=True)
    parser.add_argument("--model_name_prefix", "-m", type=str, default="")
    parser.add_argument("--model_out_dir", "-o", type=str, default="/model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_name = args.model_name_prefix + args.camera_uuid.replace("/", "_")

    with open(args.config, encoding="utf-8") as config_text:
        parsed_config = yaml.safe_load(config_text)
    parsed_config["model"]["name"] = model_name
    parsed_config["wandb"]["init_params"]["name"] = model_name
    trainer = TrainClassifier(config=parsed_config)

    # Change data path
    parsed_config["data"]["dataloader_params"]["data_path"] = args.data_path

    # Create path for saving model
    os.makedirs(args.model_out_dir, exist_ok=True)

    # Train the model
    trainer.train()

    # Test the model
    trainer.test()
