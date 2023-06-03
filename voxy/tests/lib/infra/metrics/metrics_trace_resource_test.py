import unittest

from lib.infra.metrics.metrics_trace_resource import MetricsTraceResource


class MetricsTraceResourceTest(unittest.TestCase):
    """
    Tests for metrics trace resource class
    """

    def test_create_metrics_trace_resource(self):
        self.assertIsNotNone(
            MetricsTraceResource(service_name="unit_tests", attributes={})
        )
