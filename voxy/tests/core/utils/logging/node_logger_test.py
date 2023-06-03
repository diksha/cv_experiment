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
import unittest
from typing import Any

from core.utils.logging.node_logger import node_logger


@node_logger
class ExampleNode:
    def __init__(self, config: dict) -> None:

        pass

    def process(self, thing: Any) -> int:
        return 3

    def finalize(self) -> int:
        return 6


class NodeLoggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.example_node = ExampleNode({})

    def test_logger(self) -> None:
        self.assertEqual(self.example_node.process(""), 3)

    def test_wrap(self) -> None:
        self.assertEqual(self.example_node.process(""), 3)

    def test_has_logger(self) -> None:
        self.assertTrue(hasattr(self.example_node, "logger"))

    def test_finalize(self) -> None:
        self.assertEqual(self.example_node.finalize(), 6)


if __name__ == "__main__":
    unittest.main()
