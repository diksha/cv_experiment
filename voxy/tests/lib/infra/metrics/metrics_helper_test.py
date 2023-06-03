import unittest
from unittest.mock import MagicMock, Mock

from lib.infra.metrics.metrics_helper import MetricsHelper


class TestMetricsHelper(unittest.TestCase):
    def setUp(self):
        self.otlp_meter = Mock()
        mock_oltp_meter_counter = Mock()
        self.otlp_meter.create_counter.return_value = mock_oltp_meter_counter
        self.metric_counter_names = ["test_counter_1", "test_counter_2"]
        self.metrics_helper = MetricsHelper(
            self.otlp_meter, self.metric_counter_names
        )

    def test_init(self):
        self.assertIsNotNone(self.metrics_helper.otlp_meter)
        self.assertIsNotNone(self.metrics_helper.registered_counter_metrics)
        self.assertEqual(
            len(self.metrics_helper.registered_counter_metrics),
            len(self.metric_counter_names),
        )

    def test_register_metrics(self):
        additional_metric_names = ["test_counter_3", "test_counter_4"]
        self.metrics_helper.register_metrics(additional_metric_names)
        self.assertEqual(
            len(self.metrics_helper.registered_counter_metrics),
            len(self.metric_counter_names) + len(additional_metric_names),
        )

    def test_register_metric(self):
        new_metric_name = "test_counter_5"
        self.metrics_helper.register_metric(new_metric_name)
        self.assertIn(
            new_metric_name, self.metrics_helper.registered_counter_metrics
        )

    def test_increment_counter(self):
        increment_amount = 5
        metric_name = "test_counter_1"
        counter = self.metrics_helper.registered_counter_metrics[metric_name]
        counter.add = MagicMock()

        self.metrics_helper.increment_counter(metric_name, increment_amount)
        counter.add.assert_called_once_with(increment_amount, None)

        unregistered_metric_name = "test_counter_6"
        self.metrics_helper.increment_counter(
            unregistered_metric_name, increment_amount
        )
        self.assertIn(
            unregistered_metric_name,
            self.metrics_helper.registered_counter_metrics,
        )

    def test_increment_counter_with_attributes(self):
        increment_amount = 5
        metric_name = "test_counter_1"
        counter = self.metrics_helper.registered_counter_metrics[metric_name]
        counter.add = MagicMock()

        atrs = {"key": "value"}
        self.metrics_helper.increment_counter(
            metric_name, increment_amount, atrs
        )
        counter.add.assert_called()

        unregistered_metric_name = "test_counter_7"
        self.metrics_helper.increment_counter(
            unregistered_metric_name, increment_amount
        )
        self.assertIn(
            unregistered_metric_name,
            self.metrics_helper.registered_counter_metrics,
        )
