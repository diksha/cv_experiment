import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from opentelemetry.sdk.trace.export import SpanExportResult

from lib.infra.metrics.logger_span_exporter import LoggerSpanExporter


class TestLoggerSpanExporter(unittest.TestCase):
    def test_export(self):
        # Mock ReadableSpan objects
        span1 = MagicMock()
        span2 = MagicMock()

        def custom_formatter(span: Any) -> str:
            """
            Custom formatter function that returns the span_id of the span.
            @param span: The span to format.
            @type span: Any
            @return: The formatted string.
            """
            return f"span_id: {span.span_id}"

        # Create LoggerSpanExporter with custom formatter
        exporter = LoggerSpanExporter(formatter=custom_formatter)

        with patch(
            "lib.infra.metrics.logger_span_exporter.logger"
        ) as mock_logger:
            # Export the spans and check the result
            result = exporter.export([span1, span2])
            self.assertEqual(result, SpanExportResult.SUCCESS)

            # Check if the logger was called with the correct formatted strings
            calls = mock_logger.trace.call_args_list
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0].args[0], custom_formatter(span1))
            self.assertEqual(calls[1].args[0], custom_formatter(span2))

    def test_force_flush(self):
        exporter = LoggerSpanExporter()

        # Test force_flush with default timeout
        self.assertTrue(exporter.force_flush())

        # Test force_flush with a custom timeout
        self.assertTrue(exporter.force_flush(timeout_millis=10000))
