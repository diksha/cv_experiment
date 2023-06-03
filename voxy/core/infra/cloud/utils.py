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
import os
from typing import List

from google.oauth2 import service_account

GOOGLE_SERVICE_ACCOUNT_JSON = "GOOGLE_SERVICE_ACCOUNT_JSON"


def use_service_account() -> bool:
    return bool(os.getenv(GOOGLE_SERVICE_ACCOUNT_JSON))


def get_service_account_credentials(
    scopes: List[str] = None,
) -> service_account.Credentials:
    service_account_json = os.getenv(GOOGLE_SERVICE_ACCOUNT_JSON)
    if service_account_json:
        service_account_dict = json.loads(service_account_json)
        return service_account.Credentials.from_service_account_info(
            service_account_dict,
            scopes=scopes,
        )
    return None
