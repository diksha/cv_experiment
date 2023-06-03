from dataclasses import dataclass


@dataclass
class ModelTrainingResult:
    """
    Model training result. This is the key
    interface between the model training.

    This is returned from training and passed into
    the model for evaluation
    """

    saved_local_path: str
