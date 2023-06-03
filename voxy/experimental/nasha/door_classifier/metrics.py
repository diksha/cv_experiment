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

from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import numpy as np


def calculate_topk_accuracy(y_pred, y, k = 2):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def calculate_metrics(pred, target, threshold=0.5):

    pred = np.array(pred > threshold, dtype=float)

    pred_flat = np.argmax(pred, 1)

    return {'weighted/precision': precision_score(y_true=target, y_pred=pred_flat, average='weighted'),

            'weighted/recall': recall_score(y_true=target, y_pred=pred_flat, average='weighted'),

            'weighted/f1': f1_score(y_true=target, y_pred=pred_flat, average='weighted'),

            }