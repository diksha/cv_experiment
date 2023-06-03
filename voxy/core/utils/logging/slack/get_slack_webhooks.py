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

from slack_sdk.http_retry.builtin_handlers import (
    ConnectionErrorRetryHandler,
    RateLimitErrorRetryHandler,
)
from slack_sdk.webhook import WebhookClient

from core.utils.aws_utils import get_secret_from_aws_secret_manager


def get_perception_verbose_sync_webhook(error_retry_count=1):
    perception_verbose_webhook_arn = (
        "arn:aws:secretsmanager:us-west-2:203670452561:"
        "secret:perception_verbose_webhook-94N0Bj"
    )
    perception_verbose_webhook = json.loads(
        get_secret_from_aws_secret_manager(perception_verbose_webhook_arn)
    )
    client = WebhookClient(
        url=perception_verbose_webhook["url"],
        retry_handlers=[
            ConnectionErrorRetryHandler(max_retry_count=error_retry_count),
            RateLimitErrorRetryHandler(max_retry_count=error_retry_count),
        ],
    )
    return client
