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
from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger

from core.ml.experiments.tracking.clearml_tracker import ClearMLTracker


@dataclass
class ExperimentTrackerDetails:
    project: str
    name: str


class ExperimentTracker:
    """Tracker for experiments. Wraps multiple trackers."""

    def __init__(
        self, experiment_tracker_details: ExperimentTrackerDetails
    ) -> None:
        """Initializes clearml trackers.

        Args:
            project (str): Project name
            name (str): Name of the current run
        """
        clearml_tracker = ClearMLTracker(
            experiment_tracker_details.project, experiment_tracker_details.name
        )
        self.experiment_trackers = [clearml_tracker]

    def log(
        self, prefix: str = "", message: dict = None, iteration: int = -1
    ) -> None:
        """Logs a dictionary to clearml as a scalar.

        Args:
            prefix (str, optional): Prefix for the message
            message (dict, optional): Message to log
            iteration (int, optional): Which iteration the message is for
        """
        for experiment_tracker in self.experiment_trackers:
            experiment_tracker.log(prefix, message, iteration)

    def get_details(self) -> List:
        """Gets details of tracking

        Returns:
            str: Returns the list of details from tracking
        """
        experiment_tracker_details = []
        for experiment_tracker in self.experiment_trackers:
            experiment_tracker_details.append(experiment_tracker.get_details())
        return experiment_tracker_details

    def update_parameters(self, metadata: dict) -> None:
        """Updates hyperparameters to experiment tracking

        Args:
            metadata (dict): Hyperparameters to log.
        """
        for experiment_tracker in self.experiment_trackers:
            experiment_tracker.update_parameters(metadata)

    def log_table(self, message: dict) -> None:
        """Logs a table

        Args:
            message (dict): Message to log
        """
        for experiment_tracker in self.experiment_trackers:
            experiment_tracker.log_table(message)

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
        for experiment_tracker in self.experiment_trackers:
            experiment_tracker.log_confusion_matrix(
                conf_matrix,
                title=title,
                xaxis=xaxis,
                yaxis=yaxis,
                classes=classes,
            )

    def flush(self) -> bool:
        """Flush cached reports and console outputs to backend


        Returns:
            bool: whether upload passed
        """
        flush_status = True
        for experiment_tracker in self.experiment_trackers:
            if not experiment_tracker.flush():
                logger.info(
                    f"Unable to flush from {experiment_tracker.get_details()}"
                )
                flush_status = False
        return flush_status
