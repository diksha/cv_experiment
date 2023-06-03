import numpy as np
import PIL
import torch
from loguru import logger
from pytorch_lightning import Trainer

from core.ml.algorithms.image_classification.vit_classifier import (
    ViTImageClassifier,
)
from core.ml.data.loaders.common.registry import ConverterRegistry
from core.ml.data.loaders.resources.converters import (  # trunk-ignore(pylint/W0611,flake8/F401)
    converters,
)
from core.ml.experiments.tracking.experiment_tracking import ExperimentTracker
from core.ml.training.metrics.registry import get_conf_matrix, get_metrics
from core.ml.training.models.model import Model
from core.ml.training.models.model_training_result import ModelTrainingResult
from core.ml.training.registry.registry import ModelClassRegistry
from core.perception.inference.lib.inference import InferenceModel
from core.perception.inference.transforms.registry import get_transforms
from core.structs.dataset import DataloaderType
from core.structs.dataset import Dataset as VoxelDataset


@ModelClassRegistry.register()
class ViTClassifierModel(Model):
    """
    Utility class to define how training a model happens.

    Clients must implement the `train()` and `evaluate()` methods
    """

    def __init__(self, config: dict):
        """
        Initializes the ViT Model

        Args:
            config (dict): the model parameter dictionary
        """
        self.config = config

    def train(
        self, dataset: VoxelDataset, experiment_tracker: ExperimentTracker
    ) -> ModelTrainingResult:
        """
        Creates and trains the ViT classifier model

        Args:
            dataset (VoxelDataset): the current voxel style dataset to use for training
            experiment_tracker (ExperimentTracker): the experiment tracker to use when training

        Returns:
            ModelTrainingResult: the result of training including the local model path
        """
        logger.info("Beginning training")
        dataloader = ConverterRegistry.get_dataloader(
            dataset,
            self.dataloader_type(),
            transforms=get_transforms(self.config["data"]["transforms"]),
        )
        train_dataloader = dataloader.get_training_set()
        val_dataloader = dataloader.get_validation_set()
        all_train_dataloader = dataloader.get_full_training_set()
        model = ViTImageClassifier(
            num_labels=all_train_dataloader.dataset.get_num_labels(),
            id_to_label=all_train_dataloader.dataset.get_id_to_label(),
            label_to_id=all_train_dataloader.dataset.get_label_to_id(),
            **self.config["model"]
        )
        trainer = Trainer(**self.config["training"])
        trainer.fit(model, train_dataloader, val_dataloader)
        model_path = model.save(self.config)
        # save model
        return ModelTrainingResult(saved_local_path=model_path)

    @classmethod
    def dataloader_type(cls) -> DataloaderType:
        """
        The current dataloader type

        Returns:
            DataloaderType: the dataloader format to be used when training
        """
        return DataloaderType.PYTORCH_IMAGE

    @classmethod
    def evaluate(
        cls,
        model_training_result: ModelTrainingResult,
        dataset: VoxelDataset,
        experiment_tracker: ExperimentTracker,
    ) -> dict:
        """
        Evaluates the trained model. Loads the model and config
        dictionary statelessly (using just the model artifact)
        and runs it against the training set

        Args:
            model_training_result (ModelTrainingResult): the current model training result
            dataset (VoxelDataset): the voxel style dataset that is used to train
            experiment_tracker (ExperimentTracker): the current experiment tracker

        Returns:
            dict: the dictionary of all metrics
        """
        model, config = ViTImageClassifier.load(
            model_training_result.saved_local_path
        )
        inference_model = ViTClassifierInferenceModel(model, config)

        # this should be cached if we had already downloaded during training
        dataloader = ConverterRegistry.get_dataloader(
            dataset, cls.dataloader_type()
        )
        data_loaders = {}
        data_loaders["test"] = dataloader.get_test_set()

        model_results = []
        targets = []

        for (data, label) in dataloader.get_image_label_iterator(
            data_loaders["test"].dataset
        ):
            predictions = inference_model.infer(data)
            model_results.append(predictions.detach().numpy())
            targets.extend([label])
        calculate_metrics = get_metrics("acc_f1_prec_rec")
        results = calculate_metrics(
            np.array(model_results),
            np.array(targets),
            labels=dataloader.get_label_ids(),
        )
        experiment_tracker.log_table(
            message=results,
        )

        classes = ["no_vest", "vest"]  # Sorted classes
        calulate_conf_matrix = get_conf_matrix()
        conf_matrix = calulate_conf_matrix(
            np.array(model_results), np.array(targets)
        )
        experiment_tracker.log_confusion_matrix(
            conf_matrix,
            "ViTImageClassifier",
            "Predictions",
            "Ground Truths",
            classes,
        )
        return results


class ViTClassifierInferenceModel(InferenceModel):
    """
    The model used to run classifier inference and evaluation
    """

    def __init__(self, model: torch.nn.Module, config: dict):
        """
        Initializes the vit classifier model

        Args:
            model (torch.nn.Module): the current classifier model
            config (dict): the config for the model
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.config = config
        self.transforms = get_transforms(self.config["data"]["transforms"])

    def preprocess(self, frame: PIL.Image) -> torch.tensor:
        """
        Preprocesses the model input frame

        Args:
            frame (PIL.Image): the raw input image

        Returns:
            torch.tensor: the transformed preprocessed input
        """
        return self.transforms(frame).to(self.device).unsqueeze(0)

    def postprocess(self, predictions: list) -> torch.tensor:
        """
        Post processes the model output, into a set
        of numpy predictions

        Args:
            predictions (list): the current list of model predictions

        Returns:
            torch.tensor: the pytorch tensor output
        """
        return torch.softmax(predictions[0][0], dim=0).cpu()

    def infer(self, frame: PIL.Image) -> torch.tensor:
        """
        Runs inference on a raw frame. Preprocesses the frame,
        runs it through the model, then postprocesses the frame

        Args:
            frame (PIL.Image): the current frame to run inference on

        Returns:
            torch.tensor: the postprocessed prediction
        """
        preprocessed_input = self.preprocess(frame)
        predictions = self.model(preprocessed_input)
        return self.postprocess(predictions)
