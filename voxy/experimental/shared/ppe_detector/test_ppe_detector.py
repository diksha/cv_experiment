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
import json
import os

import cv2
import numpy as np
import torch
import wandb
import yaml
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
from torchvision import ops
from transformers import DetrFeatureExtractor
from transformers.models.detr.modeling_detr import DetrObjectDetectionOutput


class SampleDataset(Dataset):
    def __init__(
        self, root_dir, num_class, img_width, img_height, label, model_type
    ):
        self._root_dir = root_dir
        self._names = [
            os.path.join(self._root_dir, label[0], file)
            for file in os.listdir(os.path.join(self._root_dir, label[0]))
            if not file.startswith(".")
        ] + [
            os.path.join(self._root_dir, label[1], file)
            for file in os.listdir(os.path.join(self._root_dir, label[1]))
            if not file.startswith(".")
        ]

        self._num_class = num_class
        self._label = {label[0]: [0], label[1]: [1]}
        self._img_width, self._img_height = img_width, img_height
        self._model_type = model_type
        if self._model_type == "DETR":
            self._feature_extractor = DetrFeatureExtractor()

    def _get_image_array(self, pil_img=None):
        if self._img_width and self._img_height:
            pil_img = pil_img.resize(
                (self._img_width, self._img_height), resample=Image.BICUBIC
            )  # resizing the images
        img_array = np.asarray(pil_img)
        img_array = img_array.transpose((2, 0, 1))
        img_array = np.ascontiguousarray(img_array)
        return torch.from_numpy(img_array).float()

    def _get_label(self, name=None):
        class_label = name.split("/")[-2]
        return torch.tensor(self._label[class_label])

    def __getitem__(self, index):
        img_file = self._names[index]
        img = Image.open(img_file)
        original_size = img.size
        input_img_data = self._get_image_array(img)

        input_label_data = self._get_label(self._names[index])
        if self._model_type == "DETR":
            encoding = self._feature_extractor(images=img, return_tensors="pt")
            # pixel_values = encoding["pixel_values"].squeeze()
            return [
                input_img_data,
                input_label_data,
                img_file,
                original_size,
                encoding,
            ]
        return [input_img_data, input_label_data, img_file, original_size]

    def __len__(self):
        return len(self._names)


class TestDetectorAsClassifier2:
    def __init__(self, config=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        self.test_dataloaders = None
        self.actor2class = {
            "PERSON": 0,
            "PIT": 1,
            "HARD_HAT": 2,
            "SAFETY_VEST": 3,
        }
        self.model = (
            torch.jit.load(self.config["model"]["path"]).float().cuda()
        )
        self.model_type = config["model"]["type"]
        if self.model_type == "DETR":
            self._feature_extractor = DetrFeatureExtractor()
        wandb.init(
            project="detector inference",
            job_type="inference",
            entity="voxel-wandb",
            tags=[self.config["experiments"]["name"]],
        )

    def _input_to_image_rectangle(self, in_put, predict, targetbbox):
        in_put = in_put.numpy().transpose((1, 2, 0))
        im = in_put.copy()
        for i in range(predict.shape[0]):
            cv2.rectangle(
                im,
                (int(predict[i, 0]), int(predict[i, 1])),
                (int(predict[i, 2]), int(predict[i, 3])),
                (255, 0, 0),
                2,
            )
        cv2.rectangle(
            im,
            (int(targetbbox[0]), int(targetbbox[1])),
            (int(targetbbox[2]), int(targetbbox[3])),
            (0, 0, 255),
            2,
        )
        return im.astype(np.uint8)

    def _calculate_metrics(self, pred, target, threshold=0.5):
        pred = np.array(pred > threshold, dtype=float)
        pred_flat = np.argmax(pred, 1)

        return {
            "metrics": precision_recall_fscore_support(
                y_true=target, y_pred=pred_flat
            ),
        }

    def load_dataset(self):
        names_bbox_files = [
            os.path.join(self.config["data"]["path"], file)
            for file in os.listdir(self.config["data"]["path"])
            if ".json" in file
        ]
        if self.config["experiments"]["cropped"]:
            test_datasets = SampleDataset(
                root_dir=self.config["data"]["path"],
                num_class=2,
                img_width=None,
                img_height=None,
                label=self.config["data"]["labels"],
                model_type=self.model_type,
            )
        else:
            test_datasets = SampleDataset(
                root_dir=self.config["data"]["path"],
                num_class=2,
                img_width=self.config["data"]["width"],
                img_height=self.config["data"]["height"],
                label=self.config["data"]["labels"],
                model_type=self.model_type,
            )
        self.test_dataloaders = torch.utils.data.DataLoader(
            test_datasets, batch_size=1, num_workers=8, shuffle=False
        )
        self.dict_bbox = {}
        for i in names_bbox_files:
            dict_current_json = open(i)
            dict_current = json.load(dict_current_json)
            self.dict_bbox.update(dict_current)

    def predict_detectron(self, _input):
        with torch.inference_mode():
            result = self.model(_input)
            return result

    def detectron_prediction_to_class_label(
        self, prediction, target_box, original_size
    ):

        boxes = prediction[0]
        pred_classes = prediction[1]
        scores = prediction[2]
        interest_boxes = boxes[
            pred_classes
            == self.actor2class[self.config["experiments"]["object"]]
        ]

        if self.config["experiments"]["cropped"]:
            y_scale = 1
            x_scale = 1
        else:
            y_scale = self.config["data"]["height"] / original_size[1]
            x_scale = self.config["data"]["width"] / original_size[0]
        target_box_converted = [
            target_box[0],
            target_box[1],
            target_box[0] + target_box[2],
            target_box[1] + target_box[3],
        ]
        target_box_converted = [
            target_box_converted[0] * x_scale,
            target_box_converted[1] * y_scale,
            target_box_converted[2] * x_scale,
            target_box_converted[3] * y_scale,
        ]
        ious = ops.box_iou(
            interest_boxes,
            torch.tensor(target_box_converted).reshape(1, -1).cuda(),
        )
        num_candidate = torch.count_nonzero(ious)
        if num_candidate == 0:
            output_guess = self.config["data"]["labels"][0]
        else:
            output_guess = self.config["data"]["labels"][1]
        return interest_boxes, target_box_converted, output_guess

    def detr_prediction_to_class_label(self, pred, target_box, original_size):
        prediction = DetrObjectDetectionOutput(
            logits=pred["logits"],
            pred_boxes=pred["pred_boxes"],
            last_hidden_state=pred["last_hidden_state"],
            encoder_last_hidden_state=pred["encoder_last_hidden_state"],
        )

        target_sizes = (
            torch.tensor(original_size[::-1]).unsqueeze(0).to(self.device)
        )
        pred = self._feature_extractor.post_process(prediction, target_sizes)
        probas = prediction.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        boxes = pred[0]["boxes"][keep]
        _ = pred[0]["scores"][keep]
        pred_classes = pred[0]["labels"][keep]

        interest_boxes = boxes[
            pred_classes
            == self.actor2class[self.config["experiments"]["object"]]
        ]
        if self.config["experiments"]["cropped"]:
            y_scale = 1
            x_scale = 1
        else:
            y_scale = self.config["data"]["height"] / original_size[1]
            x_scale = self.config["data"]["width"] / original_size[0]
        target_box_converted = [
            target_box[0],
            target_box[1],
            target_box[0] + target_box[2],
            target_box[1] + target_box[3],
        ]
        target_box_converted = [
            target_box_converted[0] * x_scale,
            target_box_converted[1] * y_scale,
            target_box_converted[2] * x_scale,
            target_box_converted[3] * y_scale,
        ]
        ious = ops.box_iou(
            interest_boxes,
            torch.tensor(target_box_converted).reshape(1, -1).cuda(),
        )
        num_candidate = torch.count_nonzero(ious)
        if num_candidate == 0:
            output_guess = self.config["data"]["labels"][0]
        else:
            output_guess = self.config["data"]["labels"][1]
        return interest_boxes.detach(), target_box_converted, output_guess

    def visualize_predictions(self):

        columns = ["Images", "Prediction", "Ground Truth"]
        for klass in self.config["data"]["labels"]:
            columns.append("score_" + klass)
        test_dt = wandb.Table(columns=columns)
        model_result = []
        targets = []
        for test_set in self.test_dataloaders:
            if self.model_type == "DETR":
                inputs, labels, name, original_size, encodings = test_set
                pixel_values = (
                    encodings["pixel_values"].squeeze(axis=0).to(self.device)
                )
                pixel_mask = (
                    encodings["pixel_mask"].squeeze(axis=0).to(self.device)
                )
                pred = self.model(
                    pixel_values=pixel_values, pixel_mask=pixel_mask
                )
            else:
                inputs, labels, name, original_size = test_set
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                pred = self.predict_detectron(inputs.squeeze(0).float())
            if self.dict_bbox:
                target_box = self.dict_bbox[name[0].split("/")[-1]]
            else:
                target_box = [0, 0, original_size[0], original_size[1]]

            if self.model_type == "DETR":
                (
                    predict_label,
                    test,
                    class_label_pred,
                ) = self.detr_prediction_to_class_label(
                    pred, target_box, original_size
                )
            else:
                (
                    predict_label,
                    test,
                    class_label_pred,
                ) = self.detectron_prediction_to_class_label(
                    pred, target_box, original_size
                )
            if class_label_pred == self.config["data"]["labels"][1]:
                model_result.extend(np.array([[0, 1]]))
            else:
                model_result.extend(np.array([[1, 0]]))
            targets.extend(labels.cpu().numpy())
            row = [
                wandb.Image(
                    self._input_to_image_rectangle(
                        inputs[0].cpu(), predict_label.cpu().numpy(), test
                    )
                ),
                class_label_pred,
                self.config["data"]["labels"][labels],
            ]
            if class_label_pred == self.config["data"]["labels"][1]:
                row.extend([0, 1])
            else:
                row.extend([1, 0])
            test_dt.add_data(*row)
        self.results = self._calculate_metrics(
            np.array(model_result), np.array(targets)
        )
        wandb.log({"image sample": test_dt})

    def log_metrics(self):
        metrics_table = wandb.Table(
            columns=["Class", "Precision", "Recall", "F1", "Support"]
        )
        for i in range(len(self.config["data"]["labels"])):
            metrics_table.add_data(
                self.config["data"]["labels"][i],
                self.results["metrics"][0][i],
                self.results["metrics"][1][i],
                self.results["metrics"][2][i],
                self.results["metrics"][3][i],
            )

        wandb.log({"test_results_detector_metrics": metrics_table})
        wandb.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as config_text:
        parsed_config = yaml.safe_load(config_text)
    test_classifier = TestDetectorAsClassifier2(config=parsed_config)
    test_classifier.load_dataset()
    test_classifier.visualize_predictions()
    test_classifier.log_metrics()
