from dataclasses import dataclass
from typing import Dict


@dataclass
class GeneratedModel:
    """
    The final result of the training manager. Consolidates the
    raw results of training and the metrics that were derived from
    the saved model
    """

    local_model_path: str
    metrics: Dict[str, object]
