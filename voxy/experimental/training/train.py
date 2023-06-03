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
import time

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from core.ml.training.dataset import Dataset
from core.ml.training.model import get_model

"""

# Train
t = Train(config_path="/mnt/disk/anurag/challenges/kits19/config.json")
t.train()
t.infer()

# Resume
t = Train(checkpoint_path="/mnt/disk/experiments/kits19_config_test/model_iter_390.pth.tar")
t.train()
t.infer()

Any part of config can be overwritten from python api.
such as config_overrides = {'experiment': {'name' : 'my_new_exp'}}
Also you can pass exp_name, it will be the final name.
"""


class Train:
    def __init__(
        self, config_path=None, checkpoint_path=None, config_overrides={}, exp_name=None
    ):
        if config_path and checkpoint_path:
            raise Exception("Config nor checkpoint path both provided")
        if config_path and os.path.exists(config_path):
            print("Training Mode")
            self.config = json.load(open(config_path))
            self.resumed = False
        elif checkpoint_path and os.path.exists(checkpoint_path):
            if "device" in config_overrides:
                map_location = torch.device(config_overrides["device"])
                self.checkpoint = torch.load(checkpoint_path, map_location=map_location)
            else:
                self.checkpoint = torch.load(checkpoint_path)
            self.config = self.checkpoint.pop("config")
            self.resumed = True
            print("Resume Mode")
        else:
            raise Exception("Neither config nor checkpoint path exists")

        self.config.update(config_overrides)
        if exp_name is not None:
            self.config["experiment"]["name"] = exp_name
        self.inference_process = []

        assert self.config["actions"]["train"], "Train False not supported yet"

        self.device = torch.device(
            self.config["device"] if torch.cuda.is_available() else "cpu"
        )

    def train(self, resume=False):
        experiment_name = self.config["experiment"]["name"]
        experiment_output_dir = os.path.join(
            self.config["experiment"]["path"], experiment_name
        )

        if not self.resumed:
            assert (
                os.path.exists(experiment_output_dir) is False
            ), "Experiment path already exists, give a new path"
            os.makedirs(experiment_output_dir)
            json.dump(
                self.config,
                open(os.path.join(experiment_output_dir, "config.json"), "w"),
                indent=4,
            )

        writer = SummaryWriter(
            logdir=os.path.join(
                self.config["experiment"]["tensorboard_path"], experiment_name
            )
        )

        train_gen, val_gen = self._create_data_loader()

        model = Train._create_model(self.config)
        model.train()
        model = model.float()
        model = model.to(self.device)

        loss_fn = self._create_loss()

        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config["train"]["start_learning_rate"],
            momentum=self.config["train"]["momentum"],
        )

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config["train"]["lr_change_epochs"],
            gamma=self.config["train"]["lr_multiplier"],
            last_epoch=-1,
        )

        if self.resumed:
            model.load_state_dict(self.checkpoint["model_dict"])
            optimizer.load_state_dict(self.checkpoint["optimizer_dict"])
            lr_scheduler.load_state_dict(self.checkpoint["lr_scheduler_dict"])

        start_epoch = self.config["train"]["start_epoch"]
        train_iter = self.config["train"]["train_iter"]

        n_batches_per_epoch = len(train_gen)

        pbar = tqdm_notebook(
            total=self.config["train"]["max_epochs"] * n_batches_per_epoch,
            initial=(self.config["train"]["start_epoch"] - 1) * n_batches_per_epoch,
        )

        for epoch in range(start_epoch, self.config["train"]["max_epochs"] + 1):
            for local_batch, local_labels in train_gen:
                pbar.set_description(
                    "Epoch: {}, Train iter: {}".format(epoch, train_iter)
                )
                pbar.update()
                time1 = time.time()

                # Zero grad so that we do not accumulate the gradients
                optimizer.zero_grad()

                local_batch = local_batch.float()
                local_labels = local_labels.float()
                # Transfer to GPU
                local_batch, local_labels = (
                    local_batch.to(self.device),
                    local_labels.to(self.device),
                )
                batch_logits = model.forward(local_batch)
                loss = loss_fn.forward(batch_logits, local_labels)
                loss.backward()
                optimizer.step()

                time2 = time.time()
                run_time = time2 - time1
                writer.add_scalar("train/loss", loss, train_iter)
                writer.add_scalar("train/batch_time", run_time, train_iter)

                if (
                    train_iter % self.config["train"]["viz_iter"] == 0
                    and train_iter != 0
                ):
                    labels_array = local_labels.cpu().detach().numpy().astype(np.uint8)
                    predictions = batch_logits

                    def _summarize_metrics(gt, pred, class_str):
                        writer.add_scalar(
                            "train/metrics/{}_precision".format(class_str),
                            precision,
                            train_iter,
                        )
                        writer.add_scalar(
                            "train/metrics/{}_recall".format(class_str),
                            recall,
                            train_iter,
                        )

                    # Call summarize metric here
                    # _summarize_metrics(gt_mask, pred_mask, 'forklift')

                    # Write image to tensorboard.
                    # writer.add_image('train/viz/overlay_gt', np.transpose(overlayed_image, (2, 0, 1)), train_iter)
                    # writer.add_image('train/viz/overlay_pred', np.transpose(pred_overlayed_image, (2, 0, 1)), train_iter)

                train_iter += 1

            if epoch % self.config["train"]["save_epoch"] == 0 and epoch != 0:
                save_path = os.path.join(
                    experiment_output_dir, "model_iter_{}.pth.tar".format(epoch)
                )
                copied_config = copy.deepcopy(self.config)
                copied_config["train"]["start_epoch"] = epoch + 1
                copied_config["train"]["train_iter"] = train_iter + 1
                state = {
                    "model_dict": model.state_dict(),
                    "optimizer_dict": optimizer.state_dict(),
                    "lr_scheduler_dict": lr_scheduler.state_dict(),
                    "config": copied_config,
                }
                torch.save(state, save_path)

            if (
                epoch % self.config["train"]["val_epoch"] == 0
                and epoch != 0
                and self.config["actions"]["val"]
            ):
                print("Running Validation")
                val_loss = 0
                val_iter = 0
                with torch.no_grad():
                    for val_local_batch, val_local_labels in val_gen:
                        # Shape and type
                        val_local_batch = val_local_batch.float()
                        val_local_labels = val_local_labels.squeeze(dim=1)
                        val_local_labels = val_local_labels.long()
                        # Transfer to GPU
                        val_local_batch, val_local_labels = (
                            val_local_batch.to(self.device),
                            val_local_labels.to(self.device),
                        )
                        val_batch_logits = model.forward(val_local_batch)
                        val_loss += loss_fn.forward(val_batch_logits, val_local_labels)
                        val_iter += 1

                    if val_iter > 0:
                        writer.add_scalar("validation/loss", val_loss / val_iter, epoch)

            lr_scheduler.step()

        writer.close()

    @staticmethod
    def _create_dataset(dataset_config):
        dataset_config = copy.deepcopy(dataset_config)
        return Dataset(**dataset_config)

    def _create_data_loader(self):
        train_gen = None
        val_gen = None
        if self.config["actions"]["train"]:
            con = copy.deepcopy(self.config["data"]["training_data_generator"])
            t_dataset = Train._create_dataset(con.pop("dataset"))
            train_gen = torch.utils.data.DataLoader(t_dataset, **con)
        if self.config["actions"]["val"]:
            con = copy.deepcopy(self.config["data"]["validation_data_generator"])
            v_dataset = Train._create_dataset(con.pop("dataset"))
            val_gen = torch.utils.data.DataLoader(v_dataset, **con)
        return train_gen, val_gen

    @staticmethod
    def _create_model(config):
        con = copy.deepcopy(config["model"])
        name = con.pop("name")
        params = con.pop("params")
        return get_model(name, params)

    def _create_loss(self):
        con = copy.deepcopy(self.config["loss"])
        if con["type"] == "cross_entropy":
            return nn.CrossEntropyLoss(
                reduction=con["params"]["reduction"],
                weight=torch.Tensor(np.array(con["params"]["weight"])).to(self.device),
            )
        if con["type"] == "focal_loss":
            return FocalLoss(
                gamma=con["params"]["gamma"],
                reduction=con["params"]["reduction"],
                class_weights=con["params"]["class_weights"],
            ).to(self.device)
        if con["type"] == "hierarchical_focal_loss":
            return HierarchicalFocalLoss(
                gammas=con["params"]["gamma"],
                reductions=con["params"]["reduction"],
                class_weights=con["params"]["class_weights"],
            ).to(self.device)

        raise Exception("Unknown Loss Type")

    # @staticmethod
    # def _class_to_color(labels_array):
    #     shp = labels_array.shape
    #     seg_color = np.zeros((shp[0], shp[1], 3), dtype=np.uint8)
    #     seg_color[np.equal(labels_array, 1)] = [255, 0, 0]
    #     seg_color[np.equal(labels_array, 2)] = [0, 0, 255]
    #     return seg_color

    # @staticmethod
    # def _overlay(features_array, seg_image, seg, alpha=0.3):
    #     # Get binary array for places where an ROI lives
    #     segbin = np.greater(seg, 0)
    #     repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    #     # Weighted sum where there's a value to overlay
    #     overlayed = np.where(
    #         repeated_segbin,
    #         np.round(alpha*seg_image+(1-alpha) *
    #                  features_array).astype(np.uint8),
    #         np.round(features_array).astype(np.uint8)
    #     )
    #     return overlayed

    # @staticmethod
    # def _run_inference_from_checkpoint(checkpoint_path, config, itype, train_iter):
    #     config = copy.deepcopy(config)
    #     checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    #     experiment_name = config['experiment']['name']
    #     experiment_output_dir = os.path.join(config['experiment']['path'], experiment_name)
    #     device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    #     model = Train._create_model(config)
    #     model = model.float()
    #     model.to(device)
    #     model.load_state_dict(checkpoint['model_dict'])

    #     infer_config = config['infer']
    #     if 'conn_comp' in infer_config['params']:
    #         conn_comp = infer_config['params'].pop('conn_comp')
    #     else:
    #         conn_comp = None

    #     if 'resample' in infer_config['params']:
    #         resample = infer_config['params'].pop('resample')
    #     else:
    #         resample = None

    #     is_hierarchical = False
    #     if config['loss']['type'] == 'hierarchical_focal_loss':
    #         is_hierarchical = True
    #     infer = Infer(model=model, device=device, is_hierarchical=is_hierarchical, **infer_config['params'])

    #     if itype == 'val':
    #         dataset = Train._create_dataset(infer_config['validation_dataset'])
    #     elif itype == 'train':
    #         dataset = Train._create_dataset(infer_config['train_dataset'])

    #     params_to_log = {'experiment_name': experiment_name, 'type': itype, 'iter': train_iter}

    #     params_to_log['is_hierarchical'] = is_hierarchical
    #     params_to_log['conn_comp'] = conn_comp
    #     params_to_log['resample'] = resample
    #     inference = Inference(infer,
    #                           dataset,
    #                           os.path.join(experiment_output_dir, "infer", str(train_iter), itype),
    #                           params_to_log,
    #                           is_hierarchical=is_hierarchical,
    #                           conn_comp=conn_comp,
    #                           resample=resample)
    #     inference.process()

    # def infer(self):
    #     print("Running Inference")
    #     assert self.config['train']['infer_epoch'] % self.config['train']['save_epoch'] == 0, "Model should be saved at each infer epoc"
    #     experiment_name = self.config['experiment']['name']
    #     experiment_output_dir = os.path.join(self.config['experiment']['path'], experiment_name)

    #     for train_epoch in range(self.config['train']['start_epoch'], self.config['train']['max_epochs'] + 1):
    #         if train_epoch % self.config['train']['infer_epoch'] == 0 and train_epoch != 0 and self.config['actions']['infer']:
    #             checkpoint_path = os.path.join(experiment_output_dir, "model_iter_{}.pth.tar".format(train_epoch))
    #             print('checkpoint_path: {}'.format(checkpoint_path))
    #             assert os.path.exists(checkpoint_path), "Checkpoint path doesn't exists"
    #             Train._run_inference_from_checkpoint(checkpoint_path, self.config, 'val', train_epoch)
    #             Train._run_inference_from_checkpoint(checkpoint_path, self.config, 'train', train_epoch)


if __name__ == "__main__":
    t = Train(
        config_path="/home/anurag_voxelsafety_com/voxel/core/ml/training/config.json"
    )
    t.train()
