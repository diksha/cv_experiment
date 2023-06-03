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
import json
import os
import pickle as pkl

import nibabel as nib
import nilearn.image
import numpy as np
from scipy.special import softmax

from .metrics import all_scores
from .metrics import logit_based_scores
from .utils import convert_to_3d_connected
from .utils import get_result


class Inference:
    def __init__(
        self,
        infer,
        dataset,
        output_path,
        params_to_log=None,
        is_hierarchical=False,
        conn_comp=None,
        output_stream=None,
        decimation_factor=None,
        resample=None,
    ):
        self.infer = infer
        self.dataset = dataset
        self.output_path = output_path
        self.params_to_log = params_to_log
        self.is_hierarchical = is_hierarchical
        self.conn_comp = conn_comp
        self.output_stream = output_stream
        self.decimation_factor = decimation_factor
        if resample is not None:
            self.resample = resample
        else:
            self.resample = {}
            self.resample["enable"] = False

    def process(self):
        scores = {}
        params = {
            "dataset_params": self.dataset.params_to_log(),
            "infer_params": self.infer.params_to_log(),
            "output_path": self.output_path,
            "input_params": self.params_to_log,
        }
        scores["params"] = params
        resampled_scores = {}
        resampled_scores["params"] = params
        # pred_logits_all = []
        # gt_all = []

        for cid in self.dataset.case_ids:
            preds = []
            # preds_logits = []
            gts = []
            for sample in self.dataset.get_samples_for_inference(cid):
                # Sample [0] is Features KHW.. [1] is Labels HW..
                (img, img_with_logits) = self.infer.process(sample[0])
                # img return would be HW..
                # img logits would be cHW.. where c is num_classes based on
                # loss.
                preds.append(img)
                # preds_logits.append(img_with_logits)
                gts.append(sample[1])
            pred = np.stack(preds, axis=0)
            gt = np.stack(gts, axis=0)
            # Converts to DcHW
            # pred_logits = np.stack(preds_logits, axis=0)
            if self.output_path is not None:
                self._dump_pred(cid, pred)

            if self.conn_comp:
                pred = convert_to_3d_connected(pred)
            scores[cid] = all_scores(pred, gt)
            # print(scores[cid])

            if self.resample["enable"]:
                original_data_path = self.resample["data_path"]
                case_path = os.path.join(original_data_path, "case_{:05d}".format(cid))

                seg = nib.load(os.path.join(case_path, "segmentation.nii.gz"))

                pred = nib.load(
                    os.path.join(
                        os.path.join(self.output_path, "case_{:05d}".format(cid)),
                        "pred.nii.gz",
                    )
                )
                resampled_pred = nilearn.image.resample_img(
                    pred, seg.affine, seg.shape, interpolation="nearest"
                )

                self._dump_resampled_pred(cid, resampled_pred)

                resampled_scores[cid] = all_scores(
                    resampled_pred.get_data(), seg.get_data()
                )

        if self.output_path is not None:
            with open(os.path.join(self.output_path, "scores.json"), "w") as fp:
                json.dump(scores, fp)

            with open(
                os.path.join(self.output_path, "resampled_scores.json"), "w"
            ) as fp:
                json.dump(resampled_scores, fp)
        else:
            self.write_to_tensorboard(scores)

        # logit_based_scores(pred_logits_all, gt_all, self.is_hierarchical, dir_path=self.output_path)

    def _dump_resampled_pred(self, cid, resampled_pred, resampled_pred_logits=None):
        dir_path = os.path.join(self.output_path, "case_{:05d}".format(cid))
        os.makedirs(dir_path, exist_ok=True)
        save_path = os.path.join(dir_path, "resampled_pred.nii.gz")
        nib.save(resampled_pred, save_path)
        if resampled_pred_logits is not None:
            np.save(
                os.path.join(dir_path, "resampled_pred_logits.npy"),
                resampled_pred_logits,
            )

    def _dump_pred(self, cid, pred, pred_logits=None):
        dir_path = os.path.join(self.output_path, "case_{:05d}".format(cid))
        os.makedirs(dir_path, exist_ok=True)
        save_path = os.path.join(dir_path, "pred.nii.gz")
        header, affine = self.dataset.get_header_and_affine(cid)
        nib.save(nib.Nifti2Image(pred, affine=affine, header=header), save_path)
        if pred_logits is not None:
            np.save(os.path.join(dir_path, "pred_logits.npy"), pred_logits)

    def write_to_tensorboard(self, scores):
        assert (
            self.output_stream is not None
        ), "No output stream provided for early inference results"
        score_types = ["dice"]
        class_str = {0: "background", 1: "kidney", 2: "tumor"}
        class_ids = [0, 1, 2]

        result = get_result(scores)
        for id in class_ids:
            for stype in score_types:
                self.output_stream.add_scalar(
                    "early_inference/metrics/{}_{}".format(class_str[id], stype),
                    result[id][stype],
                    self.params_to_log["epoch"],
                )
