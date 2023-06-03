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

from typing import Dict, Optional

from lib.infra.metrics.constants import Environment
from lib.infra.metrics.metrics_helper import MetricsHelper
from lib.infra.metrics.metrics_trace_resource import MetricsTraceResource
from lib.infra.metrics.metrics_tracer import MetricsTracer


class RegisterMetrics:
    def __init__(
        self,
        service_name: str,
        metric_names: list[str],
        attributes: dict,
        environment: Environment,
    ) -> None:
        self._tracer_resource = MetricsTraceResource(
            service_name=service_name,
            attributes=attributes,
        ).get_metrics_trace_resource()
        self._open_tracer = MetricsTracer(self._tracer_resource, environment)
        self._otlp_tracer = self._open_tracer.get_oltp_tracer()
        self._otlp_meter = self._open_tracer.get_oltp_meter()
        self._metrics_helper = MetricsHelper(self._otlp_meter, metric_names)

    def get_metrics_tracer(self):
        """
        Get the tracer used to create spans

        Returns:
            Tracer: The tracer used to create spans
        """
        return self._otlp_tracer

    def get_metrics_meter(self):
        """
        Get the meter used to create metrics

        Returns:
            Meter: The meter used to create metrics
        """
        return self._otlp_meter

    def increment_metric_counter(
        self,
        metric_name,
        count,
        attributes: Optional[Dict[str, str]] = None,
    ):
        """
        Increments the metric counter by the given count

        Args:
            metric_name (str): Name of the metric to increment
            count (int): Amount to increment the metric by
            attributes (Dict): OT Attribute /Prometheus labels
                example counter generated -> counter_test{label_key="label_value"}

        Returns:
            bool: True if the metric was incremented, False otherwise
        """
        return self._metrics_helper.increment_counter(
            metric_name=metric_name, count=count, attributes=attributes
        )
