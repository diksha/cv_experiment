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
from typing import List

from core.infra.cloud import firestore_utils


def read_from_symphony_collection(uuid: str, key: str) -> List:
    """Reads value of key from google cloud firestore symphony table for given uuid."""
    return firestore_utils.read_from_output_store("symphony", uuid, key)


def append_to_symphony_collection(uuid: str, key: str, value: List) -> None:
    """Appends value to key in google cloud firestore symphony table for given uuid atomically."""
    return firestore_utils.append_to_output_store("symphony", uuid, key, value)


def write_to_symphony_collection(uuid: str, key: str, value: List) -> None:
    """Write value to key in google cloud firestore symphony table for given uuid atomically."""
    return firestore_utils.write_to_output_store("symphony", uuid, key, value)
