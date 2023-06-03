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
import numpy as np
from clearml import Logger, Task
from loguru import logger
from retry import retry

from core.ml.experiments.tracking.base_experiment_tracker import (
    BaseExperimentTracker,
)


class ClearMLTracker(BaseExperimentTracker):
    """Tracker for experiments wrapping clearml (https://clear.ml/)

    Args:
        BaseExperimentTracker (object): Interface for Clearml Tracker
    """

    def __init__(self, project: str, name: str) -> None:
        """Initializes clearml and creates a new run

        Args:
            project (str): Project name
            name (str): Name of the current run
        """
        self.task = None
        try:
            self.task = Task.get_task(project_name=project, task_name=name)
        # trunk-ignore(pylint/W0703)
        except Exception:
            logger.warning(
                "Task not available for project and task, creating new task"
            )
        if not self.task:
            self.task = Task.init(project_name=project, task_name=name)

    @retry(TypeError, tries=3, delay=1, backoff=2)
    def _get_logger(self) -> Logger:
        """Gets the clearml logger

        Returns:
            Logger: Returns the clearml logger
        """
        return self.task.get_logger()

    def log(
        self, prefix: str = "", message: dict = None, iteration: int = -1
    ) -> None:
        """Logs a dictionary to clearml as a scalar. Scalar can be used to
        plot a scalar series when logged for different iteration.

        Args:
            prefix (str, optional): Prefix for the message
            message (dict, optional): Message to log
            iteration (int, optional): Which iteration the message is for
        """
        if prefix:
            message = {f"{prefix}_{key}": val for key, val in message.items()}
        clearml_logger = self.task.get_logger()
        for key, value in message.items():
            # Series here is a dummy name given to the plot of the scalar
            clearml_logger.report_scalar(
                title=key, series="series", iteration=iteration, value=value
            )

    def get_details(self) -> str:
        """Gets details of clearml run

        Returns:
            str: Returns the url for clearml run
        """
        return (
            "https://app.clearml.private.voxelplatform.com/projects/"
            f"{self.task.project}/experiments/{self.task.task_id}"
        )

    def update_parameters(self, metadata: dict) -> None:
        """Updates hyper parameters to clearml run

        Args:
            metadata (dict): Hyperparameters to log.
        """
        self.task.connect(metadata)

    def log_table(self, message: dict) -> None:
        """Logs a table to clearml

        Args:
            message (dict): Message to log
        """
        for key, value in message.items():
            self.task.get_logger().report_single_value(key, value)

    def log_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        title: str,
        xaxis: str,
        yaxis: str,
        classes: list,
    ) -> None:
        """Logs Confusion matrix to clearml

        Args:
            conf_matrix (np.ndarray): A heat-map matrix (example: confusion matrix)
            title (str): Title of the run
            xaxis (str): x axis label
            yaxis (str): y axis label
            classes (list): model classes
        """
        clearml_logger = self.task.get_logger()
        clearml_logger.report_confusion_matrix(
            "Confusion matrix",
            title,
            matrix=conf_matrix,
            xaxis=xaxis,
            yaxis=yaxis,
            yaxis_reversed=True,
            xlabels=classes,
            ylabels=classes,
        )

    def flush(self) -> bool:
        """Flush cached reports and console outputs to backend


        Returns:
            bool: whether upload passed
        """
        return self.task.get_logger().flush(wait=True)
