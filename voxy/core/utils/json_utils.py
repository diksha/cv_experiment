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
import re
from typing import Any, List, Tuple


class MyDict(dict):
    def __repr__(self):
        return "{" + ", ".join([f"{k}: {v!r}" for k, v in self.items()]) + "}"


def remove_double_quote_from_key_hook_fn(value: List[Tuple[str, Any]]):
    return MyDict(value)


def underscore_to_camel(name: str) -> str:
    under_pat = re.compile(r"_([a-z])")
    return under_pat.sub(lambda x: x.group(1).upper(), name)
