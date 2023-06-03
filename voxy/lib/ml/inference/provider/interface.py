from abc import abstractmethod

import torch


class AbstractInferenceProvider:
    """Abstract inference provider skeleton"""

    @abstractmethod
    def process(self, batched_input: torch.tensor) -> list:
        """Abstract process function to be implemented by child class
        Args:
            batched_input (torch.tensor): NHWC inputs (batch, height, width, channels)
        Raises:
            NotImplementedError: if the process has not been implemented in the child class
        """
        raise NotImplementedError("Inference provider must implement process")
