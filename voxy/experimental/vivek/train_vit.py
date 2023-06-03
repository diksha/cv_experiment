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

import copy
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import wandb
from loguru import logger
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    WeightedRandomSampler,
)
from torchvision import datasets
from transformers import ViTFeatureExtractor, ViTForImageClassification


class MapDataset(Dataset):
    """
    Maps the dataset to the feature extractor
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index: int) -> tuple:
        """
        Returns a single image with the feature extractor run on it and its label
        Args:
            index(int): Image index
        Returns:
            image,label (tuple): Returns a transformed image and label pair
        """
        if self.map:
            image = self.map(self.dataset[index][0])
        else:
            image = self.dataset[index][0]
        label = self.dataset[index][1]
        return image, label

    def __len__(self) -> int:
        """
        Get the length of the dataset
        Returns:
            (int): length of the dataset
        """
        return len(self.dataset)


class MergeDatasets:
    """
    Creates the different datasets for merging
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.train = None
        self.val = None
        self.dataset = None

    def create_datasets(self):
        """
        Maps dataset to feature extractor and creates train, val splits randomly
        """
        self.dataset = datasets.ImageFolder(self.root_dir)

        num_train = len(self.dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        train_orig = torch.utils.data.Subset(self.dataset, train_idx)
        val_orig = torch.utils.data.Subset(self.dataset, valid_idx)
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k", return_tesnor="pt"
        )
        self.train = MapDataset(train_orig, feature_extractor)
        self.val = MapDataset(val_orig, feature_extractor)

    def get_class_count(self):
        """
        Counts the number of elements in each class
        Returns:
            class_count(int): number of elements in each class in the entire dataset (train and val)
            train_count(int): number of elements in each class in the train set
            val_count(int): number of elements in each class in the val set
        """
        val_count = dict(
            Counter(
                list(np.array(self.dataset.targets)[self.val.dataset.indices])
            )
        )
        train_count = dict(
            Counter(
                list(
                    np.array(self.dataset.targets)[self.train.dataset.indices]
                )
            )
        )
        class_count = {
            k: val_count.get(k, 0) + train_count.get(k, 0)
            for k in set(val_count) & set(train_count)
        }
        return class_count, val_count, train_count

    def get_train_weights(self, dataset, train, train_count):
        """
        Gets the training weights of the train set to account for imbalanced classes in the dataset
        Args:
            dataset(datasets.ImageFolder): complete dataset
            train(MapDataset): training subset of dataset
            train_count(int): number of items per class in train set
        Returns:
            class_weights_train_all(list): weights for each item in train set
        """
        target_list_train = list(
            np.array(dataset.targets)[train.dataset.indices]
        )
        class_count_train = [i for _, i in sorted(train_count.items())]
        class_weights_train = 1.0 / torch.tensor(
            class_count_train, dtype=torch.float
        )
        class_weights_train_all = class_weights_train[target_list_train]
        return class_weights_train_all

    def get_samplers(self, class_weights_train_all, num_samples):
        """
        Sample the train and validation for dataloaders
        Args:
            class_weights_train_all(list): list with weights of each item in the train set
            num_samples(int): number of total items to sample with replacement
        Returns:
            weighted_sampler_train(WeightedRandomSampler): weighted random sampler for train set
            random_sampler_val(RandomSampler): random sampler for val set
        """

        weighted_sampler_train = WeightedRandomSampler(
            weights=class_weights_train_all,
            num_samples=num_samples["train"],
            replacement=True,
        )
        random_sampler_val = RandomSampler(
            data_source=self.val,
            replacement=True,
            num_samples=num_samples["val"],
        )
        return weighted_sampler_train, random_sampler_val

    def get_dataloaders(
        self,
        batch_size,
        weighted_sampler_train,
        random_sampler_val,
    ) -> dict:
        """
        Get the train and val dataloaders of a particular source dataset
        Args:
            batch_size(int): batch size of dataloader
            weighted_sampler_train(WeightedRandomSampler): weighted random sampler for train set
            random_sampler_val(RandomSampler): random sampler for val set without weighting
        Returns:
            dataloaders(dict): A dictionary of dataloaders
        """
        dataloaders = {
            "train": DataLoader(
                self.train,
                batch_size=batch_size,
                sampler=weighted_sampler_train,
                pin_memory=True,
            ),
            "val": DataLoader(
                self.val,
                batch_size=batch_size,
                sampler=random_sampler_val,
                pin_memory=True,
            ),
        }
        return dataloaders


class TrainViTImageClassifier:
    """
    Trains a ViT Image Classifier Model

    """

    def __init__(self):

        self.model_name_or_path = "google/vit-base-patch16-224-in21k"

        self.data_paths = {
            "all": "/home/vivek/voxel/vivek_cleaned_hard_hat_data/r18_10_epochs_cleaned_modesto/",
        }
        self.dataloaders = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {
            "num_epochs": 10,
            "learning_rate": 1e-4,
            "step": 10,
            "tags": "vit_laredo_walton",
            "batch_size": 32,
            "num_labels": 2,
            "device": device,
            "criterion": nn.CrossEntropyLoss(),
            "num_samples": {},
            "model_path": "/home/vivek/voxel/experimental/vivek/hard_hat_models/",
        }
        self.optimizer = None
        self.scheduler = None
        self.model = None

    def get_dataloader(self):
        """
        Get train and val dataloaders for all datasets from all sources
        Returns:
            _id2label(dict): maps integer id to class name
            _label2id(dict): maps class name to integer id
        """
        dataset_dict = {}
        dataloaders = {}
        source_datasets = {}
        for source, root_dir in self.data_paths.items():
            source_datasets[source] = MergeDatasets(root_dir)
            source_dataset = source_datasets[source]
            source_dataset.create_datasets()
            if len(source_dataset.train) > self.config["num_samples"].get(
                "train", 0
            ):
                self.config["num_samples"]["val"] = len(source_dataset.val)
                self.config["num_samples"]["train"] = len(source_dataset.train)
            dataset_dict[source] = {
                "dataset": source_dataset.dataset,
                "train": source_dataset.train,
                "val": source_dataset.val,
            }

        for source in self.data_paths:
            source_dataset = source_datasets[source]

            _, _, train_count = source_dataset.get_class_count()
            class_weights_train_all = source_dataset.get_train_weights(
                source_dataset.dataset, source_dataset.train, train_count
            )

            (
                weighted_sampler_train,
                random_sampler_val,
            ) = source_dataset.get_samplers(
                class_weights_train_all, self.config["num_samples"]
            )

            dataloaders[source] = source_dataset.get_dataloaders(
                self.config["batch_size"],
                weighted_sampler_train,
                random_sampler_val,
            )
        for phase in ["train", "val"]:
            self.dataloaders[phase] = []
            for _, dataloader in dataloaders.items():
                self.dataloaders[phase].append(dataloader[phase])

        self.config["num_labels"] = len(source_dataset.dataset.classes)
        _id2label = {
            str(i): c for i, c in enumerate(source_dataset.dataset.classes)
        }
        _label2id = {
            c: str(i) for i, c in enumerate(source_dataset.dataset.classes)
        }

        print(f"\n\n_id2label: {_id2label}")
        print(f"_label2id: {_label2id}")

        return _id2label, _label2id

    def get_model(self, _id2label, _label2id):
        """
        Initialize model, optimizer and scheduler for training
        """

        self.model = ViTForImageClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=self.config["num_labels"],
            id2label=_id2label,
            label2id=_label2id,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=self.config["step"], gamma=0.1
        )

    def _evaluate_epoch(self, phase: str, running_metrics: dict):
        """Evaluate a single epochs metrics

        Args:
            phase (str): train or val phase
            running_metrics (dict): dict of all running metrics

        Returns:
            dict: aggregated metrics for epoch
        """
        epoch_metrics = {}
        epoch_metrics["epoch_loss"] = (
            running_metrics["running_loss"] / self.config["num_samples"][phase]
        )
        epoch_metrics["epoch_acc"] = (
            running_metrics["running_corrects"].double()
            / self.config["num_samples"][phase]
        )

        epoch_metrics["epoch_precision"] = running_metrics["running_tp"] / (
            running_metrics["running_tp"] + running_metrics["running_fp"]
        )
        epoch_metrics["epoch_recall"] = running_metrics["running_tp"] / (
            running_metrics["running_tp"] + running_metrics["running_fn"]
        )
        epoch_metrics["epoch_f1"] = (
            2
            * (
                epoch_metrics["epoch_precision"]
                * epoch_metrics["epoch_recall"]
            )
            / (
                epoch_metrics["epoch_precision"]
                + epoch_metrics["epoch_recall"]
            )
        )

        return epoch_metrics

    def train_model(self):
        """
        Runs training using model and dataset specified

        """

        wandb.init(
            project="vit_classification",
            entity="voxel-wandb",
            config=self.config,
            tags=[self.config["tags"]],
        )

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_f1 = 0
        self.model.to(self.config["device"])
        today_date = datetime.today().strftime("%Y-%m-%d")

        for epoch in range(self.config["num_epochs"]):
            logger.debug(
                f"Epoch {epoch+1}/{self.config['num_epochs']} \n \
                    ----------------"
            )

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_metrics = {
                    "running_loss": 0,
                    "running_corrects": 0,
                    "running_tp": 0,
                    "running_fp": 0,
                    "running_fn": 0,
                }
                for inputs_labels in zip(*self.dataloaders[phase]):

                    inputs = torch.cat(
                        [
                            _input[0]["pixel_values"][0]
                            for _input in inputs_labels
                        ],
                        0,
                    )
                    labels = torch.cat([_label[1] for _label in inputs_labels])

                    inputs = inputs.to(self.config["device"])
                    labels = labels.to(self.config["device"]).view(-1)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        outputs = outputs.logits
                        _, preds = torch.max(outputs, 1)
                        loss = self.config["criterion"](outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()
                    # statistics
                    running_metrics[
                        "running_loss"
                    ] += loss.item() * inputs.size(0)
                    running_metrics["running_corrects"] += torch.sum(
                        preds == labels.data
                    )
                    running_metrics["running_tp"] += torch.sum(
                        np.logical_and(preds.cpu(), labels.data.cpu())
                    ).cuda()
                    running_metrics["running_fp"] += torch.sum(
                        np.greater(preds.cpu(), labels.data.cpu())
                    ).cuda()
                    running_metrics["running_fn"] += torch.sum(
                        np.less(preds.cpu(), labels.data.cpu())
                    ).cuda()

                if phase == "train":
                    self.scheduler.step()
                epoch_metrics = self._evaluate_epoch(phase, running_metrics)

                if phase == "val":

                    wandb.log({"val loss": epoch_metrics["epoch_loss"]})
                    wandb.log({"val_f1": epoch_metrics["epoch_f1"]})
                if phase == "train":
                    wandb.log({"train loss": epoch_metrics["epoch_loss"]})

                logger.debug(
                    f"{phase} Loss: {epoch_metrics['epoch_loss']:.4f} \
                        Acc: {epoch_metrics['epoch_acc']:.4f} \
                            Precision: {epoch_metrics['epoch_precision']:.4f} \
                                Recall: {epoch_metrics['epoch_recall']:.4f} \
                                    F1: {epoch_metrics['epoch_f1']:.4f}"
                )

                # deep copy the model
                if phase == "val" and epoch_metrics["epoch_f1"] > best_f1:
                    best_f1 = epoch_metrics["epoch_f1"]
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        logger.debug(f"Training complete \n Best val Acc: {best_f1:4f}")

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        # save this model

        torch.save(
            self.model.state_dict(),
            f"{self.config['model_path']}/voxel_safetyvest_{self.config['tags']}_{today_date}.pth",
        )
        wandb.join()


if __name__ == "__main__":
    trainViT = TrainViTImageClassifier()
    id2label, label2id = trainViT.get_dataloader()
    trainViT.get_model(id2label, label2id)
    trainViT.train_model()
