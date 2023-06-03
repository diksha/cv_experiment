import unittest

from core.execution.utils.perception_runner_context import (
    PerceptionRunnerContext,
)


class ContextUnitTest(unittest.TestCase):
    def test_get_triton_server_url_from_context(self):
        ctx = PerceptionRunnerContext("192.168.0.1:8000")
        result = ctx.triton_server_url
        self.assertEqual(result, "192.168.0.1:8000")

    def test_get_default_triton_server_url(self):
        ctx = PerceptionRunnerContext()
        result = ctx.triton_server_url
        self.assertEqual(result, "127.0.0.1:8001")
