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
import os
import unittest

from mock import MagicMock, patch

from core.infra.cloud import utils


class UtilsTest(unittest.TestCase):
    @patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_JSON": ""})
    def test_use_service_account_false(self) -> None:
        self.assertEqual(utils.use_service_account(), False)

    @patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_JSON": "{}"})
    def test_use_service_account_true(self) -> None:
        self.assertEqual(utils.use_service_account(), True)

    @patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_JSON": ""})
    def test_get_service_account_credentials_None(self) -> None:
        self.assertEqual(utils.get_service_account_credentials(), None)

    @patch.dict(os.environ, {"GOOGLE_SERVICE_ACCOUNT_JSON": "{}"})
    @patch("core.infra.cloud.utils.service_account")
    def test_get_service_account_credentials(
        self, mock_service_account: MagicMock
    ) -> None:
        mock_service_account.Credentials.from_service_account_info.return_value = (
            {}
        )
        print(utils.get_service_account_credentials())
        self.assertEqual(utils.get_service_account_credentials(), {})


if __name__ == "__main__":
    unittest.main()
