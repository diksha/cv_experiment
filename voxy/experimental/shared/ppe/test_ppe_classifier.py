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
import os

import numpy as np
import torch
import wandb
from sklearn.metrics import precision_recall_fscore_support
from torchvision import datasets, transforms


class TestClassifier:
    TEST_TABLE_NAME = "test_results_images"
    RESULT_TABLE_NAME = "test_results_metrics"
    PROJECT_NAME = "door_classification"

    def __init__(
        self,
        labels=["no_vest", "vest"],
        model_path="artifacts_vest_classifier-10-26/voxel_vest_classifier_dataset3_resnet50_traced_model.pth",
        model_name="production_vest_classifier",
        test_dir="/home/nasha_voxelsafety_com/voxel/experimental/shared/ppe/scenario_test",
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.MODEL_NAME = model_name
        self.TEST_DIR = test_dir
        self.model = torch.jit.load(model_path).eval().float().cuda()
        self.class_labels = labels
        self.results = None
        self.test_dataloaders = None
        wandb.init(
            project=self.PROJECT_NAME,
            job_type="inference",
            entity="voxel-wandb",
        )

    def _input_to_image(self, in_put):
        in_put = in_put.numpy().transpose(
            (1, 2, 0)
        )  # reshape the input to be like image m by n by 3
        in_put = np.clip(in_put, 0, 1)  # clip the values out of [0,1] band
        in_put = (in_put * 255).astype(
            np.uint8
        )  # multiply by 255 (pixels are in range of 0 to 255)
        return in_put

    def _calculate_metrics(self, pred, target, threshold=0.5):
        pred = np.array(pred > threshold, dtype=float)
        pred_flat = np.argmax(pred, 1)

        return {
            "metrics": precision_recall_fscore_support(
                y_true=target, y_pred=pred_flat
            ),
        }

    def load_dataset(self):

        data_transforms = {
            "test": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            ),
        }

        test_datasets = {
            x: datasets.ImageFolder(self.TEST_DIR, data_transforms[x])
            for x in ["test"]
        }

        self.test_dataloaders = {
            x: torch.utils.data.DataLoader(
                test_datasets[x], batch_size=1, num_workers=4, shuffle=False
            )
            for x in ["test"]
        }

    def visualize_predictions(self):
        columns = ["Images", "Prediction", "Ground Truth"]
        for klass in self.class_labels:
            columns.append("score_" + klass)
        test_dt = wandb.Table(columns=columns)
        model_result = []
        targets = []
        for inputs, labels in self.test_dataloaders["test"]:

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            pred = self.model(inputs.type(torch.float32))

            model_result.extend(pred.cpu().detach().numpy())
            targets.extend(labels.cpu().numpy())

            pred = pred.data.cpu().numpy()
            guess = self.class_labels[np.argmax(pred)]

            row = [
                wandb.Image(self._input_to_image(inputs[0].cpu())),
                guess,
                self.class_labels[labels],
            ]

            for c_i in pred[0].tolist():
                row.append(np.round(c_i, 4))
            test_dt.add_data(*row)
        self.results = self._calculate_metrics(
            np.array(model_result), np.array(targets)
        )
        wandb.log({self.TEST_TABLE_NAME: test_dt})

    def log_metrics(self):
        metrics_table = wandb.Table(
            columns=["Class", "Precision", "Recall", "F1", "Support"]
        )

        for i in range(len(self.class_labels)):
            metrics_table.add_data(
                self.class_labels[i],
                self.results["metrics"][0][i],
                self.results["metrics"][1][i],
                self.results["metrics"][2][i],
                self.results["metrics"][3][i],
            )

        wandb.log({self.RESULT_TABLE_NAME: metrics_table})
        wandb.join()


if __name__ == "__main__":
    test_classifier = TestClassifier(
        labels=["closed", "open", "partially_open"],
        model_path="/home/nasha_voxelsafety_com/voxel/experimental/nasha/models/door_classifier/door_classifier_best-2022-04-27-bloomingdale-model-base-EMA-jit.pth",
        # test_dir="/home/nasha_voxelsafety_com/voxel/experimental/nasha/images/val/",
        test_dir="/home/nasha_voxelsafety_com/datasets/bloomingdale/test/",
    )
    # load the dataset
    test_classifier.load_dataset()
    # run model on dataset and upload visuals to wandb
    test_classifier.visualize_predictions()
    # log the overall metrics of the classifier
    test_classifier.log_metrics()
