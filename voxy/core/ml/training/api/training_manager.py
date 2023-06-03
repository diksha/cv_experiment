import core.ml.training.registry.register_components  # trunk-ignore(pylint/W0611,flake8/F401)
from core.ml.common.torch_utils import initialize_random_state
from core.ml.data.generation.common.pipeline import DatasetMetaData
from core.ml.experiments.tracking.experiment_tracking import ExperimentTracker
from core.ml.training.api.generated_model import GeneratedModel
from core.ml.training.registry.registry import ModelClassRegistry
from core.structs.dataset import Dataset as VoxelDataset


class TrainingManager:
    """
    Training manager is responsible for training the model and providing
    any metadata that could be useful to keep track of in the model registry

    Responsibilities include:
     1. Construct a model
     2. Train a model
     3. Evaluate the model metrics using test dataset
     4. Return saved model (local)
    """

    DEFAULT_RANDOM_STATE = 1234

    def __init__(
        self,
        model_config: dict,
        experiment_tracker: ExperimentTracker,
    ):
        """
        Initializes the training manager with a model config and any experiment trackers

        Args:
            model_config (dict): the config used to generate the model
            experiment_tracker (ExperimentTracker): the experiment tracker used to
                                track an experiment
        """
        self.model = ModelClassRegistry.get_instance(**model_config)
        self.experiment_tracker = experiment_tracker
        self.model_config = model_config

    def config(self) -> dict:
        """
        Returns the config used for the model when training

        Returns:
            dict: the dictionary used by the model to configure training
        """
        return self.model_config

    def download_dataset(
        self, dataset: VoxelDataset, dataset_metadata: DatasetMetaData
    ) -> "TrainingManager":
        """Downloads dataset

        Args:
            dataset (VoxelDataset): the dataset to train with
            dataset_metadata (DatasetMetaData): Metadata of the dataset

        Returns:
            self: Training Mananager
        """
        dataset.download(dataset_metadata.local_path)
        return self

    def train(
        self: "TrainingManager",
        dataset: VoxelDataset,
    ) -> GeneratedModel:
        """
        Trains the model. Returns the trained model

        Args:
            dataset (VoxelDataset): the dataset to train with

        Returns:
            (GeneratedModel): the result of model training containing
                                     both the model training result and the metrics

        """
        initialize_random_state(self.DEFAULT_RANDOM_STATE)
        model_training_result = self.model.train(
            dataset, self.experiment_tracker
        )
        metrics = self.model.evaluate(
            model_training_result, dataset, self.experiment_tracker
        )
        return GeneratedModel(
            local_model_path=model_training_result.saved_local_path,
            metrics=metrics,
        )
