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
"""
This is a test file created to test bazel is working.
You can run it using bazel run //experimental/anurag/test
"""
# trunk-ignore-all(pylint)
# trunk-ignore-all(flake8)

import os
import sqlite3
import sys
import uuid
from sqlite3.dbapi2 import *

import attr
import cython
import shapely
from _sqlite3 import *

print(sys.executable)

print(uuid.uuid4())
print(cython.__version__)
print(os.listdir("artifacts_doors_0630/"))
