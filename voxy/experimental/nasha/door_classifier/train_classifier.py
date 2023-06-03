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
from datetime import datetime

import numpy as np
import torch
import wandb
import yaml
from lib.datasets.registry import get_dataset
from lib.models.classifiermodels import ModelEMA
from lib.models.registry import get_model_deprecated
from lib.transforms.registry import get_transforms
from metrics import calculate_metrics, calculate_topk_accuracy
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms


class TrainClassifier:
    def __init__(self, config):
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

    def test(self):
        data_loaders, dataset_sizes = self._create_dataloaders()
        # load model
        self.model.load_state_dict(
            torch.load(f'{self.config["test"]["model_path"]}')
        )
        test_loss, test_acc_1, test_acc_2, test_f1 = self._evaluate_epoch(
            iterator=data_loaders["test"]
        )
        print(
            f"Test Loss: {test_loss:.3f} | Test Acc @1: {test_acc_1*100:6.2f}% | "
            f"Test Acc @2: {test_acc_2*100:6.2f}%"
        )

    def train(self):
        data_loaders, dataset_sizes = self._create_dataloaders()

        # scheduling
        scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[p["lr"] for p in self.optimizer.param_groups],
            total_steps=self.config["optimizer"]["n_epoch"]
            * dataset_sizes["train"],
        )

        ema_model = ModelEMA(self.model)
        for epoch in range(self.config["optimizer"]["n_epoch"]):
            print(f"Epoch: {epoch+1:02}")

            train_loss, train_acc_1, train_acc_5 = self._train_epoch(
                ema_model, iterator=data_loaders["train"], scheduler=scheduler
            )

            wandb.log({"train loss": train_loss})
            print(
                f"\tTrain Acc @1: {train_acc_1*100:6.2f}% | "
                f"Train Acc @2: {train_acc_5*100:6.2f}%"
            )

            (
                valid_loss,
                valid_acc_1,
                valid_acc_5,
                val_f1,
            ) = self._evaluate_epoch(iterator=data_loaders["val"])

            wandb.log({"val loss": valid_loss})
            wandb.log({"val_f1": val_f1})
            today_date = datetime.today().strftime("%Y-%m-%d")
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(
                    self.model.state_dict(),
                    f'{self.config["model_path"]}/door_classifier_best-{today_date}-{os.path.basename(self.config["data"]["dataloader_params"]["data_path"])}-model-base-EMA.pt',
                )

            best_model_wts = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(best_model_wts)
            example_input = torch.randn(
                1, 3, 224, 224, requires_grad=False
            ).to(self.device)
            traced_model = torch.jit.trace(self.model, example_input)
            traced_model.save(
                f'{self.config["model_path"]}/door_classifier_best-{today_date}-{os.path.basename(self.config["data"]["dataloader_params"]["data_path"])}-model-base-EMA-jit.pth'
            )

            print(
                f"\t Valid Loss: {valid_loss:.3f} | Valid Acc @1: {valid_acc_1*100:6.2f}% | "
                f"Valid Acc @2: {valid_acc_5*100:6.2f}%"
            )

    def _evaluate_epoch(self, iterator):

        epoch_loss = 0
        epoch_acc_1 = 0
        epoch_acc_5 = 0
        model_resultt = []
        targetss = []
        self.model.eval()

        with torch.no_grad():

            for (x, y) in iterator:

                x = x.to(self.device)
                print(x.shape)
                y = y.to(self.device)
                y_pred = self.model(x)

                loss = self.criterion(y_pred, y)

                acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc_1 += acc_1.item()
                epoch_acc_5 += acc_5.item()
                model_resultt.extend(y_pred.cpu().detach().numpy())
                targetss.extend(y.cpu().numpy())
        results = calculate_metrics(
            np.array(model_resultt), np.array(targetss)
        )

        print(
            f"\t val_Precision: {results['weighted/precision']:.4f} val_Recall: {results['weighted/recall']:.4f} val_F1: {results['weighted/f1']:.4f}"
        )

        epoch_loss /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_5 /= len(iterator)

        return epoch_loss, epoch_acc_1, epoch_acc_5, results["weighted/f1"]

    def _train_epoch(self, ema_model, iterator, scheduler):

        epoch_loss = 0
        epoch_acc_1 = 0
        epoch_acc_5 = 0
        model_result = []
        targets = []
        self.model.train()
        for (x, y) in iterator:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)

            acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

            loss.backward()
            ema_model.update(self.model)

            self.optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
            model_result.extend(y_pred.cpu().detach().numpy())
            targets.extend(y.cpu().numpy())
        epoch_loss /= len(iterator)
        epoch_acc_1 /= len(iterator)
        epoch_acc_5 /= len(iterator)
        results = calculate_metrics(np.array(model_result), np.array(targets))

        print(
            f" Train Loss: {epoch_loss:.4f} Precision: {results['weighted/precision']:.4f} Recall: {results['weighted/recall']:.4f} F1: {results['weighted/f1']:.4f}"
        )

        return epoch_loss, epoch_acc_1, epoch_acc_5

    def _calculate_norms(self):
        data_dir = self.config["data"]["dataloader_params"]["data_path"]
        train_dir = os.path.join(data_dir, "train")
        dataset_loader = get_dataset(self.config["data"]["dataloader_name"])
        train_data = dataset_loader(
            root=train_dir, transform=transforms.ToTensor()
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

        train_data = image_data["train"]
        n_train_examples = int(len(train_data) * (1 - params["valid_pct"]))
        n_valid_examples = len(train_data) - n_train_examples

        image_data["train"], image_data["val"] = data.random_split(
            train_data, [n_train_examples, n_valid_examples]
        )
        return image_data

    def _create_dataloaders(self):
        con = copy.deepcopy(self.config["data"])

        if not con["normalize"]:
            mean, std = self._calculate_norms()
        else:
            mean, std = con["normalize"]["mean"], con["normalize"]["std"]

        name = con.pop("dataloader_name")
        params = con.pop("dataloader_params")
        dataset_class = get_dataset(name)
        data_transforms = {}
        image_data = {}
        for phase in ["train", "test"]:
            con[f"{phase}_transform"]["item_tfms"].append(
                {"name": "Normalize", "params": {"mean": mean, "std": std}}
            )
            data_transforms[phase] = con.pop(f"{phase}_transform")
            data_transforms[phase] = get_transforms(
                data_transforms[phase].pop("item_tfms", [])
            )

            image_data[phase] = dataset_class(
                root=os.path.join(params["data_path"], phase),
                transform=data_transforms[phase],
            )

        # split the train into valid/train
        image_data = self._random_split(params, image_data)

        dataloaders = {}
        dataset_sizes = {}
        weighted_samplers = {}
        for phase in ["train", "val", "test"]:
            shuffle = False  # True if phase == "train" else False

            target = np.array(image_data[phase])[:, 1]

            class_sample_count = np.array(
                [len(np.where(target == t)[0]) for t in np.unique(target)]
            )
            weight = 1.0 / class_sample_count
            samples_weight = np.array([weight[t] for t in target])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            weighted_sampler = WeightedRandomSampler(
                samples_weight, len(samples_weight)
            )
            weighted_samplers[phase] = (
                weighted_sampler if phase != "test" else None
            )

            dataloaders[phase] = data.DataLoader(
                image_data[phase],
                shuffle=shuffle,
                batch_size=params["batch_size"],
                sampler=weighted_samplers[phase],
            )

            dataset_sizes[phase] = len(image_data[phase])

        return dataloaders, dataset_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as config_text:
        parsed_config = yaml.safe_load(config_text)
    trainer = TrainClassifier(config=parsed_config)
    if parsed_config["test"]["model_path"] is not None:
        trainer.test()
    else:
        trainer.train()
