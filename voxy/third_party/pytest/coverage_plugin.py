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

import coverage
from coverage.control import Coverage


class CoveragePlugin:
    def __init__(self, coverage_output_path):

        # Monkey Patch, such that it uses the relative file paths
        # rather than absolute.
        coverage.collector.abs_file = CoveragePlugin.abs_file
        coverage.files.abs_file = CoveragePlugin.abs_file
        coverage.control.abs_file = CoveragePlugin.abs_file

        self.coverage_output_path = coverage_output_path
        self.coverage_runner = Coverage(
            source=["core", "lib", "services"],
            omit=[
                "*_test.py",
                "*_pb2.py",
                "core/utils/mkvtagreader/mkv_element_ids.py",
            ],
            branch=True,
        )
        self.coverage_runner.start()

    def pytest_sessionfinish(self, *args, **kwargs):
        self.coverage_runner.stop()
        self.coverage_runner.lcov_report(outfile=self.coverage_output_path)

    @staticmethod
    def abs_file(filename):
        return coverage.files.actual_path(
            os.path.abspath(os.path.expandvars(os.path.expanduser(filename)))
        )


def pytest_configure(config):
    # This variable is expected to be set by bazel coverage.
    if os.getenv("COVERAGE_OUTPUT_FILE"):
        plugin = CoveragePlugin(os.getenv("COVERAGE_OUTPUT_FILE"))
        config.pluginmanager.register(plugin)
