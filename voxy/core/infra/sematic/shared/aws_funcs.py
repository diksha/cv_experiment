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
"""Utility functions for using with Sematic & AWS"""
from enum import Enum, unique
from typing import Optional

import sematic


@unique
class PathKind(Enum):
    """The kind of S3 path that's expected"""

    DIRECTORY = "DIRECTORY"
    FILE = "FILE"
    FILE_OR_DIRECTORY = "FILE_OR_DIRECTORY"


# trunk-ignore-begin(pylint/W9015,pylint/W9011)
@sematic.func
def get_bucket_from_s3_uri(full_path: Optional[str]) -> str:
    """# Given an S3 URI, get the bucket

    ## Parameters
    - **full_path**:
        An s3 URI like s3://my-bucket/my/path

    ## Returns
    The bucket name
    """
    # trunk-ignore-end(pylint/W9015,pylint/W9011)
    return full_path.replace("s3://", "").split("/")[0]
