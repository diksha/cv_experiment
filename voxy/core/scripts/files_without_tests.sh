#!/bin/bash
##
## Copyright 2020-2021 Voxel Labs, Inc.
## All rights reserved.
##
## This document may not be reproduced, republished, distributed, transmitted,
## displayed, broadcast or otherwise exploited in any manner without the express
## prior written permission of Voxel Labs, Inc. The receipt or possession of this
## document does not convey any rights to reproduce, disclose, or distribute its
## contents, or to manufacture, use, or sell anything that it may describe, in
## whole or in part.
##

set -euo pipefail

ALL_PYTHON_SOURCE_FILES=$(find core -type f -name '*.py' -a ! -name '*_test.py' | sort)
ALL_FILES_WITH_COVERAGE=$(awk '/^SF:/' bazel-out/_coverage/_coverage_report.dat | awk '-FSF:' '{print $2}' | sort)
diff --side-by-side --suppress-common-lines <(echo "$ALL_PYTHON_SOURCE_FILES") <(echo "$ALL_FILES_WITH_COVERAGE") || true
