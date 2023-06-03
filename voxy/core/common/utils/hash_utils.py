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

import hashlib
import json


def dictionary_hash(dictionary: dict) -> str:
    """
    Creates a hash of a dictionary by creating a
    json string that is sorted by the keys and
    finds the sha256 of the resulting string

    Args:
        dictionary (dict): the dictionary to hash

    Returns:
        str: the string of the hash
    """
    return str(
        hashlib.sha256(
            json.dumps(dictionary, sort_keys=True).encode("utf-8")
        ).hexdigest()
    )
