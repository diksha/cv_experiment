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
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2, reduction="weighted_mean", class_weights=[1.0, 1.0, 1.0]
    ):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._reduction = reduction
        self._class_weights = class_weights
        self._softmax = nn.Softmax2d()
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets, pixel_weights=None):
        loss = self._cross_entropy_loss(logits, targets)
        loss = loss.view(loss.size(0), -1)

        targets = targets.view(targets.size(0), -1)
        targets = targets.unsqueeze(dim=1)

        probs = self._softmax(logits)
        probs = probs.view(probs.size(0), probs.size(1), -1)
        probs = probs.gather(dim=1, index=targets)
        prob_weights = (1 - probs).pow(self._gamma)

        class_weights = torch.zeros_like(targets, dtype=torch.float32)
        for idx in range(len(self._class_weights)):
            class_weights[targets == idx] = self._class_weights[idx]

        total_weight = prob_weights
        total_weight *= class_weights
        if pixel_weights is not None:
            pixel_weights = pixel_weights.view(pixel_weights.size(0), -1)
            pixel_weights = pixel_weights.unsqueeze(dim=1)
            total_weight *= pixel_weights

        if self._reduction == "weighted_mean":
            return (total_weight * loss).sum() / (total_weight.sum() + 1e-6)
        raise ValueError("Given Reduction: {} is not supported".format(self._reduction))


class HierarchicalFocalLoss(nn.Module):
    def __init__(
        self,
        gammas=[2, 2],
        reductions=["weighted_mean", "weighted_mean"],
        class_weights=[[1.0, 1.0], [1.0, 1.0]],
    ):
        super(HierarchicalFocalLoss, self).__init__()

        self._focal_loss1 = FocalLoss(
            gamma=gammas[0], reduction=reductions[0], class_weights=class_weights[0]
        )
        self._focal_loss2 = FocalLoss(
            gamma=gammas[1], reduction=reductions[1], class_weights=class_weights[1]
        )

    def forward(self, logits, targets):
        # The first loss is for foreground/background
        targets1 = (targets > 0).long()
        logits1 = logits[:, :2, ...]
        weights1 = torch.ones_like(targets1, dtype=torch.float32)
        loss1 = self._focal_loss1(logits1, targets1.long(), weights1)

        # The second loss is for class2/class3
        logits2 = logits[:, 2:, ...]
        weights2 = (targets > 0).float()
        targets2 = torch.zeros_like(targets1)
        targets2[targets == 1] = 0
        targets2[targets == 2] = 1
        loss2 = self._focal_loss1(logits2, targets2.long(), weights2)

        return loss1 + loss2


class SmoothL1Loss(nn.Module):
    "Smooth L1 Loss"

    def __init__(self, beta=0.11):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target):
        x = (pred - target).abs()
        l1 = x - 0.5 * self.beta
        l2 = 0.5 * x ** 2 / self.beta
        return torch.where(x >= self.beta, l1, l2)
