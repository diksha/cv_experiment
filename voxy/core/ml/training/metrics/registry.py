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

from fastai.metrics import accuracy, accuracy_multi

from core.ml.training.metrics.metrics import (
    calculate_confusion_matrix,
    calculate_metrics,
)

# trunk-ignore-all(pylint/W9011)
# trunk-ignore-all(pylint/C0116)
# trunk-ignore-all(pylint/W9012)

REGISTRY = {
    "accuracy_multi": accuracy_multi,
    "accuracy": accuracy,
    "acc_f1_prec_rec": calculate_metrics,
}


def get_metrics(name):
    return REGISTRY[name]


def get_conf_matrix():
    return calculate_confusion_matrix
