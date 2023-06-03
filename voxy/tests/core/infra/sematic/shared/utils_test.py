#
# Copyright 2023 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.

import argparse
import unittest
from unittest.mock import MagicMock, patch

from sematic.future import Future

from core.infra.sematic.shared.utils import (
    SematicOptions,
    resolve_sematic_future,
)


class SematicOptionsTest(unittest.TestCase):
    def test_option_parser(self):
        """Tests that option parse can add and parse options"""
        parser = argparse.ArgumentParser()
        SematicOptions.add_to_parser(parser)
        args = parser.parse_args([])
        obj = SematicOptions.from_args(args)
        self.assertIsInstance(obj, SematicOptions)


class ResolveSematicFutureTest(unittest.TestCase):
    @patch("core.infra.sematic.shared.utils.CloudResolver")
    @patch("core.infra.sematic.shared.utils.LocalResolver")
    @patch("core.infra.sematic.shared.utils.SilentResolver")
    @patch("core.infra.sematic.shared.utils.has_container_image")
    def test_cloud_resolver(
        self,
        has_container_image,
        silent_resolver,
        local_resolver,
        cloud_resolver,
    ):
        """Test that we run cloud resolver when we have a container image"""
        has_container_image.return_value = True
        mock_sematic_options = MagicMock(spec_set=SematicOptions)
        mock_future = MagicMock(spec=Future, id=MagicMock())

        resolve_sematic_future(mock_future, mock_sematic_options, "label")

        mock_future.resolve.assert_called_once()

        local_resolver.assert_not_called()
        silent_resolver.assert_not_called()
        cloud_resolver.assert_called_once_with(
            cache_namespace=mock_sematic_options.cache_namespace,
            rerun_from=mock_sematic_options.rerun_from,
            max_parallelism=mock_sematic_options.max_parallelism,
        )

    @patch("core.infra.sematic.shared.utils.CloudResolver")
    @patch("core.infra.sematic.shared.utils.LocalResolver")
    @patch("core.infra.sematic.shared.utils.SilentResolver")
    @patch("core.infra.sematic.shared.utils.has_container_image")
    def test_local_resolver(
        self,
        has_container_image,
        silent_resolver,
        local_resolver,
        cloud_resolver,
    ):
        """Test that we run local resolver when we don't have a container image"""
        has_container_image.return_value = False
        mock_sematic_options = MagicMock(spec_set=SematicOptions)
        mock_sematic_options.silent = False
        mock_future = MagicMock(spec=Future, id=MagicMock())

        resolve_sematic_future(mock_future, mock_sematic_options, "label")

        mock_future.resolve.assert_called_once()

        cloud_resolver.assert_not_called()
        silent_resolver.assert_not_called()
        local_resolver.assert_called_once_with(
            cache_namespace=mock_sematic_options.cache_namespace,
            rerun_from=mock_sematic_options.rerun_from,
        )

    @patch("core.infra.sematic.shared.utils.CloudResolver")
    @patch("core.infra.sematic.shared.utils.LocalResolver")
    @patch("core.infra.sematic.shared.utils.SilentResolver")
    @patch("core.infra.sematic.shared.utils.has_container_image")
    def test_silent_resolver(
        self,
        has_container_image,
        silent_resolver,
        local_resolver,
        cloud_resolver,
    ):
        """
        Test that we run silent resolver when we don't have a container
        image and we have the silent option
        """
        has_container_image.return_value = False
        mock_sematic_options = MagicMock(spec_set=SematicOptions)
        mock_sematic_options.silent = True
        mock_future = MagicMock(spec=Future, id=MagicMock())

        resolve_sematic_future(mock_future, mock_sematic_options, "label")

        mock_future.resolve.assert_called_once()

        cloud_resolver.assert_not_called()
        local_resolver.assert_not_called()
        silent_resolver.assert_called_once_with()
