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

import json

from core.common.utils.recursive_namespace import RecursiveSimpleNamespace


class Datapool(RecursiveSimpleNamespace):
    """
    Datapool metadata wrapper used when grabbing from metaverse
    in the graphql response
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = (
            json.loads(kwargs["lightly_config"])
            if "lightly_config" in kwargs
            else None
        )

    def get_config(self) -> dict:
        """
        Grabs the config from the datapool graphql response

        Returns:
            dict: the lightly configuration
        """
        return self.config
