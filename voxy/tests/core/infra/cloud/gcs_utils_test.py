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
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from google.cloud import storage

from core.infra.cloud.gcs_utils import download_files


class GCSUtilsTest(unittest.TestCase):
    @patch("core.infra.cloud.gcs_utils.storage.Client")
    @patch("core.infra.cloud.gcs_utils.storage.Bucket")
    def test_download_files(
        self, mock_bucket: MagicMock, mock_gcs_client: MagicMock
    ):
        blob = storage.Blob(
            name="relative_path",
            bucket=mock_bucket,
        )

        mock_gcs_client.return_value.lookup_bucket.return_value.return_value.list_blobs = [
            blob
        ]
        with tempfile.TemporaryDirectory() as tempdir:
            assert download_files("gs://bucket/relative_path", tempdir)
            assert os.path.exists(tempdir)
