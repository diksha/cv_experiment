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
from abc import ABC, abstractmethod

import numpy as np


class BaseExperimentTracker(ABC):
    """Base for experiment Tracker

    Args:
        ABC: Abstract base class
    """

    @abstractmethod
    def __init__(self, project: str, name: str) -> None:
        """Base for experiment tracker

        Args:
            project (str): Project name
            name (str): Name of the current run
        """

    @abstractmethod
    def log(
        self, prefix: str = "", message: dict = None, iteration: int = -1
    ) -> None:
        """Abstract method for logging scalar values.

        Args:
            prefix (str, optional): Prefix for the message
            message (dict, optional): Message to log
            iteration (int, optional): Which iteration the message is for
        """

    @abstractmethod
    def get_details(self) -> str:
        """Gets details of trackers

        Returns:
            str: Returns the url for trackers
        """

    @abstractmethod
    def update_parameters(self, metadata: dict) -> None:
        """Updates hyper parameters

        Args:
            metadata (dict): Hyperparameters to log.
        """

    @abstractmethod
    def log_table(self, message: dict) -> None:
        """Logs a table

        Args:
            message (dict): Message to log
        """

    @abstractmethod
    def log_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        title: str,
        xaxis: str,
        yaxis: str,
        classes: list,
    ) -> None:
        """Logs a Confusion Matrix

        Args:
            conf_matrix (np.ndarray): confusion matrix between target and pred labels
            title (str): Title of the run
            xaxis (str): x axis label
            yaxis (str): y axis label
            classes (list): model classes
        """

    @abstractmethod
    def flush(self) -> bool:
        """Flush cached reports and console outputs to backend


        Returns:
            bool: whether flush completed successfully
        """
