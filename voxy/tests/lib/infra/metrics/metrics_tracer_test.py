import unittest
from unittest.mock import patch

from opentelemetry.sdk.resources import Resource

from lib.infra.metrics.constants import Environment
from lib.infra.metrics.metrics_tracer import MetricsTracer


class MetricsTracerTest(unittest.TestCase):
    """
    Tests for metrics tracer
    """

    def setUp(self):
        self.resource = Resource({"key": "value"})
        self.environment = Environment.DEVELOPMENT

    @patch("lib.infra.metrics.metrics_tracer.OTLPSpanExporter")
    @patch("lib.infra.metrics.metrics_tracer.LoggerSpanExporter")
    @patch("lib.infra.metrics.metrics_tracer.BatchSpanProcessor")
    @patch("lib.infra.metrics.metrics_tracer.TracerProvider")
    @patch("lib.infra.metrics.metrics_tracer.OTLPMetricExporter")
    @patch("lib.infra.metrics.metrics_tracer.PeriodicExportingMetricReader")
    @patch("lib.infra.metrics.metrics_tracer.MeterProvider")
    def test_get_exporter(
        self,
        meter_provider_mock,
        periodic_exporting_metric_reader_mock,
        otlp_metric_exporter_mock,
        tracer_provider_mock,
        batch_span_processor_mock,
        logger_span_exporter_mock,
        otlp_span_exporter_mock,
    ):
        MetricsTracer(self.resource, self.environment)
        self.assertEqual(logger_span_exporter_mock.call_count, 1)

        MetricsTracer(self.resource, Environment.PRODUCTION)
        self.assertEqual(otlp_span_exporter_mock.call_count, 1)
