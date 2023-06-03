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

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

# trunk-ignore-all(pylint/C0116)
# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(pylint/W9012)


def calculate_metrics(pred, target, labels, threshold=0.5, k=2):
    pred_flat = np.argmax(pred, 1)

    return {
        "top_1_accuracy": accuracy_score(y_true=target, y_pred=pred_flat),
        "top_k_accuracy": top_k_accuracy_score(
            y_true=target,
            y_score=pred if len(labels) > 2 else pred_flat,
            k=k,
            labels=labels,
        ),
        "weighted/precision": precision_score(
            y_true=target, y_pred=pred_flat, average="weighted"
        ),
        "weighted/recall": recall_score(
            y_true=target, y_pred=pred_flat, average="weighted"
        ),
        "weighted/f1": f1_score(
            y_true=target, y_pred=pred_flat, average="weighted"
        ),
    }


def calculate_confusion_matrix(pred, target):
    """Computes the confusion matrix

    Args:
        pred (_type_): predicted classes from model
        target (_type_): ground truth classes

    Returns:
        _type_: sklearn np.array matrix
    """
    pred_flat = np.argmax(pred, 1)
    return confusion_matrix(y_true=target, y_pred=pred_flat)
