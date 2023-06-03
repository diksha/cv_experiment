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

from collections import defaultdict
from typing import Dict, List, Optional

from opentelemetry.metrics import Meter


class MetricsHelper:
    """
    A helper class for managing metrics and tracing.

    Args:
        otlp_meter (Meter): The meter used to create metrics.
        metric_counter_names (List[str]): A list of metric names to register with the meter.
    """

    def __init__(
        self, otlp_meter: Meter, metric_counter_names: List[str]
    ) -> None:
        self.otlp_meter = otlp_meter
        self.registered_counter_metrics = defaultdict()

        self.register_metrics(metric_counter_names)

    def register_metrics(self, metric_names: List[str]) -> None:
        """
        Registers metrics with the provided names.

        Args:
            metric_names (List[str]): A list of metric names to register.
        """
        for metric_name in metric_names:
            self.register_metric(metric_name)

    def register_metric(self, metric_name: str):
        """
        Registers a metric with the provided name.
        Args:
            metric_name (str): Name of the metric to register.
        """
        self.registered_counter_metrics[
            metric_name
        ] = self.otlp_meter.create_counter(metric_name)

    def increment_counter(
        self,
        metric_name: str,
        count: int,
        attributes: Optional[Dict[str, str]] = None,
    ):
        """
        Increments a counter by 1

        Args:
            metric_name (str): Name of the metric to increment
            count (int): Amount to increment the metric by
            attributes (Dict): OT Attribute /Prometheus labels
                ex counter generated -> counter_test{label_key="label_value"}
        """
        if metric_name not in self.registered_counter_metrics:
            self.register_metric(metric_name)
        self.registered_counter_metrics[metric_name].add(count, attributes)
