from datetime import datetime

import yaml

from core.ml.training.api.training_manager import TrainingManager
from core.utils.yaml_jinja import resolve_jinja_config

if __name__ == "__main__":
    train_metadata_yaml = "core/ml/training/configs/door_training_new.yaml"
    train_metadata = yaml.safe_load(resolve_jinja_config(train_metadata_yaml))
    train_metadata["model_parameters"][
        "name"
    ] = f'door_training_{datetime.today().strftime("%Y-%m-%d")}'
    TrainingManager(train_metadata, "experiment_run").run()
