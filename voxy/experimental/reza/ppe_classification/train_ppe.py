#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.

import argparse
import copy
import os
import time
from datetime import datetime

import numpy as np
import torch
import wandb
import yaml
from lib.datasets.registry import get_dataset
from lib.models.registry import get_model_deprecated
from lib.transforms.registry import get_transforms
from loss import FocalLoss
from torch import nn
from torch.optim import lr_scheduler
from utils import get_data_loader


# trunk-ignore-all(pylint/R0915)
# trunk-ignore-all(pylint/C0209)
# trunk-ignore-all(pylint/W1514)
class Train:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            self.config["device"] if torch.cuda.is_available() else "cpu"
        )

        wandb.init(**self.config["wandb"]["init_params"])

    def train(self):
        model, optimizer, scheduler = self._create_model_optimizer_scheduler()
        criterion = self._create_criterion()
        data_loaders, dataset_sizes = self._create_dataloaders()
        self._learn(
            model, data_loaders, dataset_sizes, optimizer, scheduler, criterion
        )

    def _learn(
        self,
        model,
        dataloaders,
        dataset_sizes,
        optimizer,
        scheduler,
        criterion,
    ):
        config = wandb.config
        model_path = self.config["save_path"]
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        today_date = datetime.today().strftime("%Y-%m-%d")
        type_classifier = self.config["classifier"]["name"]
        model_name = str(self.config["model"]["name"])
        loss_name = (
            str(self.config["loss"]["name"])
            if self.config["loss"]["use_cross_entropy"] == 0
            else "CE"
        )
        wighted_sampling = (
            "WS"
            if self.config["data"]["dataloader_params"][
                "use_weighted_sampling"
            ]
            == 1
            else "NWS"
        )
        param_loss = (
            str(self.config["loss"]["params"]["gamma"])
            if loss_name == "focal_loss"
            else ""
        )
        model_file_name = f"{model_path}/voxel_{type_classifier}_classifier_{model_name}_{loss_name}_gamma{param_loss}_{wighted_sampling}_{today_date}.pth"
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_f1 = 0
        for epoch in range(config.num_epochs):
            print("Epoch {}/{}".format(epoch, config.num_epochs - 1))
            print("-" * 10)
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                running_tp = 0
                running_fp = 0
                running_fn = 0
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device).view(-1)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_tp += torch.sum(
                        np.logical_and(preds.cpu(), labels.data.cpu())
                    ).cuda()
                    running_fp += torch.sum(
                        np.greater(preds.cpu(), labels.data.cpu())
                    ).cuda()
                    running_fn += torch.sum(
                        np.less(preds.cpu(), labels.data.cpu())
                    ).cuda()

                if phase == "train":
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_precision = running_tp / (running_tp + running_fp)
                epoch_recall = running_tp / (running_tp + running_fn)
                epoch_f1 = (
                    2
                    * (epoch_precision * epoch_recall)
                    / (epoch_precision + epoch_recall)
                )
                if phase == "val":
                    print("val loss", epoch_loss)
                    print("val_f1", epoch_f1)
                    wandb.log({"val loss": epoch_loss})
                    wandb.log({"val_f1": epoch_f1})
                if phase == "train":
                    print("train loss", epoch_loss)
                    wandb.log({"train loss": epoch_loss})

                print(
                    "{} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
                        phase,
                        epoch_loss,
                        epoch_acc,
                        epoch_precision,
                        epoch_recall,
                        epoch_f1,
                    )
                )
                if phase == "val" and epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_f1))

        with open("./best_f1.txt", "w") as f:
            f.write(str(float(best_f1)))
        f.close()
        model.load_state_dict(best_model_wts)
        example_input = torch.randn(1, 3, 224, 224, requires_grad=False).to(
            self.device
        )
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(model_file_name)
        wandb.join()
        return model

    def _create_model_optimizer_scheduler(self):
        con = copy.deepcopy(self.config["model"])
        name = con.pop("name")
        model_params = con.pop("params")
        model_class = get_model_deprecated(name)
        model = model_class(**model_params)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.config["optimizer"]["lr"], momentum=0.9
        )
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["scheduler"]["step"],
            gamma=self.config["scheduler"]["gamma"],
        )
        model.train()
        model = model.to(self.device)
        return model, optimizer, scheduler

    def _create_criterion(self):
        if self.config["loss"]["use_cross_entropy"]:
            loss = nn.CrossEntropyLoss()
        else:
            loss = FocalLoss(
                self.config["loss"]["params"]["alpha"],
                self.config["loss"]["params"]["gamma"],
                reduction=self.config["loss"]["params"]["reduction"],
            )
        return loss

    def _create_dataloaders(self):
        con = copy.deepcopy(self.config["data"])
        name = con.pop("dataloader_name")
        params = con.pop("dataloader_params")
        dataset_class = get_dataset(name)
        dataset = dataset_class(params["path"])
        data_transforms = {}
        for phase in ["train", "val"]:
            data_transforms[phase] = con.pop(f"{phase}_transform")
            data_transforms[phase] = get_transforms(
                data_transforms[phase].pop("item_tfms", [])
            )
        data_loaders, dataset_sizes, class_names = get_data_loader(
            dataset=dataset,
            percentage=params["valid_pct"],
            batch_size=params["bs"],
            weighted_sampling=params["use_weighted_sampling"],
            data_transforms=data_transforms,
        )
        return data_loaders, dataset_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as config_text:
        parsed_config = yaml.safe_load(config_text)
    trainer = Train(config=parsed_config)
    trainer.train()
