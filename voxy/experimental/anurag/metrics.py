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
from bokeh.plotting import ColumnDataSource
from bokeh.plotting import figure
from bokeh.plotting import output_file
from bokeh.plotting import save
from sklearn.metrics import precision_recall_curve


def confusion_scores(pred, gt, class_id):
    TP = np.sum((pred == class_id) & (gt == class_id))
    TN = np.sum((pred != class_id) & (gt != class_id))
    FP = np.sum((pred == class_id) & (gt != class_id))
    FN = np.sum((pred != class_id) & (gt == class_id))
    return TP, TN, FP, FN


def multiclass_confusion_matrix(pred, gt, class_ids):
    cm = np.zeros((len(class_ids), len(class_ids)))
    for c1 in class_ids:
        for c2 in class_ids:
            cm[c1][c2] = np.sum((gt == c1) & (pred == c2))
    return cm


def precision_from_confusion(TP, TN, FP, FN):
    if (TP + FP) == 0:
        return 0
    return (1.0 * TP) / (TP + FP)


def recall_from_confusion(TP, TN, FP, FN):
    if (TP + FN) == 0:
        return 0
    return (1.0 * TP) / (TP + FN)


def iou_from_confusion(TP, TN, FP, FN):
    if (TP + FP + FN) == 0:
        return 0
    else:
        return (1.0 * TP) / (TP + FP + FN)


def dice_from_confusion(TP, TN, FP, FN):
    denominator = (2.0 * TP) + FP + FN
    if denominator == 0.0:
        return 0.0
    else:
        return (2.0 * TP) / denominator


def all_scores_for_class(pred, gt, class_id):
    class_scores = {}
    conf = confusion_scores(pred, gt, class_id)
    class_scores["TP"] = int(conf[0])
    class_scores["TN"] = int(conf[1])
    class_scores["FP"] = int(conf[2])
    class_scores["FN"] = int(conf[3])
    class_scores["dice"] = dice_from_confusion(*conf)
    class_scores["precision"] = precision_from_confusion(*conf)
    class_scores["recall"] = recall_from_confusion(*conf)
    class_scores["iou"] = iou_from_confusion(*conf)
    return class_scores


def dice_score(pred, gt, class_ids=[0, 1, 2]):
    assert isinstance(
        pred, np.ndarray
    ), "pred array should be np.ndarray. Provided: {}".format(type(pred))
    assert isinstance(
        gt, np.ndarray
    ), "gt array should be np.ndarray. Provided: {}".format(type(pred))
    assert (
        pred.shape == gt.shape
    ), "pred and gt shape should be same. Provided pred shape: {}, gt shape: {}".format(
        pred.shape, gt.shape
    )
    return {
        class_id: dice_from_confusion(*confusion_scores(pred, gt, class_id))
        for class_id in class_ids
    }


def all_scores(pred, gt, class_ids=[0, 1, 2]):
    assert isinstance(
        pred, np.ndarray
    ), "pred array should be np.ndarray. Provided: {}".format(type(pred))
    assert isinstance(
        gt, np.ndarray
    ), "gt array should be np.ndarray. Provided: {}".format(type(gt))
    assert (
        pred.shape == gt.shape
    ), "pred and gt shape should be same. Provided pred shape: {}, gt shape: {}".format(
        pred.shape, gt.shape
    )
    scores = {
        class_id: all_scores_for_class(pred, gt, class_id) for class_id in class_ids
    }
    scores["confusion_multiclass"] = multiclass_confusion_matrix(
        pred, gt, class_ids
    ).tolist()
    scores["confusion_multiclass_sum_of_row"] = "Actual"
    scores["confusion_multiclass_sum_of_column"] = "Pred"
    scores["shape"] = list(gt.shape)
    return scores


def logit_based_scores(pred_logits, gt, is_hierarchical, dir_path, class_ids=[1, 2]):
    if not is_hierarchical:
        scores = {}
        for class_id in class_ids:
            gts = []
            preds = []
            for p, g in zip(pred_logits, gt):
                gt_class = np.zeros(g.shape)
                gt_class[np.where(g == class_id)] = 1
                # It's DCHW, therefore get the channel which is class id.
                pred_class = p[:, class_id]
                gts.append(gt_class.flatten())
                preds.append(pred_class.flatten())
            precision, recall, threshold = precision_recall_curve(
                np.concatenate(gts).ravel(), np.concatenate(preds).ravel()
            )
            threshold = threshold + [0] * (len(precision) - len(threshold))
            output_file(os.path.join(dir_path, "Class_{}.html".format(class_id)))
            source = ColumnDataSource(
                data=dict(
                    recall=list(recall),
                    precision=list(precision),
                    threshold=list(threshold),
                )
            )

            TOOLTIPS = [
                ("(recall, precision)", "($recall, $precision)"),
                ("threshold", "@threshold"),
            ]

            p = figure(tooltips=TOOLTIPS, title="Class: {}".format(class_id))
            p.line("recall", "precision", source=source)
            save(p)
    else:
        scores = {}

        # Kidney
        gts = []
        preds = []
        for p, g in zip(pred_logits, gt):
            gt_class = np.zeros(g.shape)
            gt_class[np.where(g == 1)] = 1
            # It's DCHW, therefore get the channel which is class id.
            pred_class = p[:, 2]
            gts.append(gt_class.flatten())
            preds.append(pred_class.flatten())

        precision, recall, threshold = precision_recall_curve(
            np.concatenate(gts).ravel(), np.concatenate(preds).ravel()
        )
        threshold = threshold + [0] * (len(precision) - len(threshold))
        output_file(os.path.join(dir_path, "Class_{}.html".format(2)))
        source = ColumnDataSource(
            data=dict(
                recall=list(recall),
                precision=list(precision),
                threshold=list(threshold),
            )
        )

        TOOLTIPS = [
            ("(recall, precision)", "($recall, $precision)"),
            ("threshold", "@threshold"),
        ]

        p = figure(tooltips=TOOLTIPS, title="Class: {}".format(2))
        p.line("recall", "precision", source=source)
        save(p)

        # Tumor
        gts = []
        preds = []
        for p, g in zip(pred_logits, gt):
            gt_class = np.zeros(g.shape)
            gt_class[np.where(g == 2)] = 1
            # It's DCHW, therefore get the channel which is class id.
            pred_class = p[:, 3]
            gts.append(gt_class.flatten())
            preds.append(pred_class.flatten())

        precision, recall, threshold = precision_recall_curve(
            np.concatenate(gts).ravel(), np.concatenate(preds).ravel()
        )
        threshold = threshold + [0] * (len(precision) - len(threshold))
        output_file(os.path.join(dir_path, "Class_{}.html".format(3)))
        source = ColumnDataSource(
            data=dict(
                recall=list(recall),
                precision=list(precision),
                threshold=list(threshold),
            )
        )

        TOOLTIPS = [
            ("(recall, precision)", "($recall, $precision)"),
            ("threshold", "@threshold"),
        ]

        p = figure(tooltips=TOOLTIPS, title="Class: {}".format(3))
        p.line("recall", "precision", source=source)
        save(p)

        # Foreground
        gts = []
        preds = []
        for p, g in zip(pred_logits, gt):
            gt_class = np.zeros(g.shape)
            gt_class[np.where(g == 1)] = 1
            gt_class[np.where(g == 2)] = 1
            # It's DCHW, therefore get the channel which is class id.
            pred_class = p[:, 1]
            gts.append(gt_class.flatten())
            preds.append(pred_class.flatten())

        precision, recall, threshold = precision_recall_curve(
            np.concatenate(gts).ravel(), np.concatenate(preds).ravel()
        )
        threshold = threshold + [0] * (len(precision) - len(threshold))
        output_file(os.path.join(dir_path, "Class_{}.html".format(1)))
        source = ColumnDataSource(
            data=dict(
                recall=list(recall),
                precision=list(precision),
                threshold=list(threshold),
            )
        )

        TOOLTIPS = [
            ("(recall, precision)", "($recall, $precision)"),
            ("threshold", "@threshold"),
        ]

        p = figure(tooltips=TOOLTIPS, title="Class: {}".format(1))
        p.line("recall", "precision", source=source)
        save(p)
