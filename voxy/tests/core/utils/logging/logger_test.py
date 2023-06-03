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

from core.utils.logging.logger import Logger


class LoggerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enabled_logger = Logger("enabled", {"log_key": "something"})
        self.disabled_logger = Logger("disabled", {})

    def test_defaults(self) -> None:
        self.assertTrue(self.enabled_logger.enabled)
        self.assertTrue(not self.disabled_logger.enabled)


if __name__ == "__main__":
    unittest.main()
