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
#

from dataclasses import dataclass


@dataclass(frozen=True)
class PerceptionRunnerContext:
    """
    Context is a dataclass that contains all the information needed to run

    Attributes:
        _triton_server_url: the url of the triton server
    """

    _triton_server_url: str = "127.0.0.1:8001"

    @property
    def triton_server_url(self) -> str:
        """
        Name of the triton server url

        Returns:
            str: the triton server url
        """
        return self._triton_server_url
