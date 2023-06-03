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

import typing
from os import linesep
from typing import Optional

from loguru import logger
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class LoggerSpanExporter(SpanExporter):
    """Implementation of :class:`SpanExporter` that prints spans to the
    logger object.

    This class can be used for diagnostic purposes. It prints the exported
    spans to the console logger.
    """

    def __init__(
        self,
        service_name: Optional[str] = None,
        formatter: typing.Callable[
            [ReadableSpan], str
        ] = lambda span: span.to_json()
        + linesep,
    ):
        self.formatter = formatter
        self.service_name = service_name

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Exports the spans to the logger.
            @param spans: The spans to export.
            @return: The result of the export.
        """
        for span in spans:
            logger.trace(self.formatter(span))
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Forces to export all finished spans.
            @param timeout_millis: The maximum amount of time to wait for spans to
            be exported.
            @return: True if all spans were exported successfully.
        """
        return True
