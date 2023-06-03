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

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.aggregation import (
    AggregationTemporality,
)
from opentelemetry.sdk.metrics._internal.instrument import Counter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from lib.infra.metrics.constants import Environment
from lib.infra.metrics.logger_metric_exporter import LoggerMetricExporter
from lib.infra.metrics.logger_span_exporter import LoggerSpanExporter


class MetricsTracer:
    """
    This class is used to create a tracer and meter provider for the metrics and tracing.
    Args:
        resource (Resource): Resource to be used for the metrics and tracing
        environment (Environment): Environment to be used for the metrics and tracing
    """

    def __init__(self, resource: Resource, environment: Environment) -> None:
        self._span_exporter = self._get_span_exporter(environment)
        self._span_processor = BatchSpanProcessor(self._span_exporter)
        self._otlp_trace_provider = TracerProvider(resource=resource)
        self._otlp_trace_provider.add_span_processor(self._span_processor)
        trace.set_tracer_provider(self._otlp_trace_provider)

        # Sets up OTLP metrics
        self._otlp_metrics_exporter = self._get_metrics_exporter(environment)
        self._otlp_metrics_reader = PeriodicExportingMetricReader(
            self._otlp_metrics_exporter
        )
        self._otlp_metrics_provider = MeterProvider(
            metric_readers=[self._otlp_metrics_reader], resource=resource
        )
        metrics.set_meter_provider(self._otlp_metrics_provider)

        self._otlp_tracer = trace.get_tracer(__name__)
        self._otlp_meter = self._otlp_metrics_provider.get_meter(__name__)

    def _get_metrics_exporter(self, environment: Environment):
        if environment == Environment.PRODUCTION:
            return OTLPMetricExporter(
                preferred_temporality={Counter: AggregationTemporality.DELTA}
            )
        return LoggerMetricExporter(
            preferred_temporality={Counter: AggregationTemporality.DELTA}
        )

    def _get_span_exporter(self, environment: Environment):
        if environment == Environment.PRODUCTION:
            return OTLPSpanExporter()
        return LoggerSpanExporter()

    def get_oltp_tracer(self):
        """
        Get the OpenTelemetry tracer for the metrics and tracing
        Returns:
            Tracer: The tracer for the metrics and tracing
        """
        return self._otlp_tracer

    def get_oltp_meter(self):
        """
        Get the OpenTelemetry meter for the metrics and tracing
        Returns:
            Meter: The meter for the metrics and tracing
        """
        return self._otlp_meter
