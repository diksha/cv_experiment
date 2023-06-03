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
from torch import nn, tensor


class ViTUncertaintyModel:
    # Beta is a hyper parameter that controls the "strength" of the
    # uncertainty that the model produces. A higher beta biases the model
    # to be less certain. Beta must be greater than 1, otherwise the dirichlet
    # distribution breaks down
    BETA = 10.0

    def __init__(self, base_vit_model, calibrated_temperature=10.0):
        self.model = base_vit_model
        self.logits_to_evidence = nn.Softplus()
        self.temperature = calibrated_temperature
        self.softmax = nn.Softmax(dim=1)

    def convert_logits_to_evidence(self, logits: tensor) -> tensor:
        """
        Converts the raw outputs of the model into the model evidence. The evidence
        must be strictly positive, this should correspond to how the model was trained

        Args:
            logits (tensor): the raw output of the model of size: (batch, n_classes)

        Returns:
            tensor: the evidence of the model based on the activation function chosen
        """
        return self.logits_to_evidence(logits)

    def convert_evidence_to_uncertainty(self, evidence: tensor) -> tuple:
        """
        Converts the raw logits of the model into the class uncertainty and the
        epistemic uncertainty based on the paper: https://arxiv.org/pdf/1806.01768.pdf

        Args:
            evidence (tensor): the raw output evidence tensor of size: (batch, n_classes)

        Returns:
            tuple: class probabilities and the raw model uncertainty
        """
        # Convert the logits to uncertainty based on equation 1 of
        # https://arxiv.org/pdf/1806.01768.pdf
        batch, n_classes = evidence.size()
        K = n_classes
        alpha = evidence + self.BETA
        S = torch.sum(alpha, dim=1, keepdims=True)
        uncertainty = (K * self.BETA) / S
        return uncertainty

    def logits_to_calibrated_probabilities(self, logits: tensor) -> tensor:
        """
        Converts the raw logits to calibrated probabilities based on the temperature provided
        at construction

        The method is based on this paper: https://arxiv.org/pdf/1706.04599.pdf

        Args:
            logits (tensor): the raw logits of the model

        Returns:
            tensor: the scaled probabilites based on the input temperature
        """
        # evaluate
        scaled_logits = logits / self.temperature
        return self.softmax(scaled_logits)

    def __call__(self, image_batch: tensor) -> tuple:
        """
        Main entry point of the uncertain ViT model.

        Args:
            image_batch (tensor): the torch tensor of size

        Returns:
            tuple: the class probabilities and epistemic uncertainty of the model based on the paper:
                   https://arxiv.org/pdf/1806.01768.pdf
        """
        logits = self.model(image_batch).logits
        evidence = self.logits_to_evidence(logits)
        epistemic_uncertainty = self.convert_evidence_to_uncertainty(evidence)
        class_probabilities = self.logits_to_calibrated_probabilities(logits)
        return class_probabilities, epistemic_uncertainty
