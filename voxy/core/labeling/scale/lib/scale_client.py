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

import scaleapi

from core.utils.aws_utils import get_secret_from_aws_secret_manager


def get_scale_client(
    credentials_arn: str,
) -> scaleapi.ScaleClient:
    """Return instantiated Scale client

    Args:
        credentials_arn (typing.Optional[str]): the credentials arn to load from secrets

    Returns:
        scaleapi.ScaleClient: connected, authenticated scale client
    """
    scale_creds = json.loads(
        get_secret_from_aws_secret_manager(credentials_arn)
    )
    return scaleapi.ScaleClient(scale_creds["api_key"])
