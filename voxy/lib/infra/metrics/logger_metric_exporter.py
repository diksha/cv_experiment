# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#

from os import linesep
from typing import Callable, Dict

from loguru import logger
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    MetricExporter,
    MetricExportResult,
    MetricsData,
)
from opentelemetry.sdk.metrics.view import Aggregation


class LoggerMetricExporter(MetricExporter):
    """Implementation of :class:`MetricExporter` that prints metrics to the
    console.

    This class can be used for diagnostic purposes. It prints the exported
    metrics to the loguru logger.
    """

    def __init__(
        self,
        formatter: Callable[
            [MetricsData], str
        ] = lambda metrics_data: metrics_data.to_json()
        + linesep,
        preferred_temporality: Dict[type, AggregationTemporality] = None,
        preferred_aggregation: Dict[type, Aggregation] = None,
    ):
        super().__init__(
            preferred_temporality=preferred_temporality,
            preferred_aggregation=preferred_aggregation,
        )
        self.formatter = formatter

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,
        **kwargs,
    ) -> MetricExportResult:
        """Logs the metrics data to the logger.

        Args:
            metrics_data (MetricsData): data to log
            timeout_millis (float, optional): unused. Defaults to 10_000.
            kwargs: unused

        Returns:
            MetricExportResult: SUCCESS
        """
        logger.trace(self.formatter(metrics_data))
        return MetricExportResult.SUCCESS

    def shutdown(self, timeout_millis: float = 30_000, **kwargs) -> None:
        pass

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        """Flushes the exporter.

        Args:
            timeout_millis (float, optional): Unused. Defaults to 10_000.

        Returns:
            bool: True
        """
        return True
