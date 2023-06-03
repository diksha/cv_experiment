import unittest

import torch

from core.perception.vest.vest_classifier import VestClassifier


class VestClassifierTest(unittest.TestCase):
    def test_postprocess_predictions(self):
        model_output = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        expected_output = torch.tensor([True, False])
        post_processed = VestClassifier.postprocess_predictions(
            model_output, {"NO_VEST": 0, "VEST": 1}
        )
        self.assertTrue(len(post_processed) == len(model_output))
        self.assertTrue(torch.all(post_processed == expected_output))
