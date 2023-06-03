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

from django.conf import settings
from google.cloud import pubsub_v1
from google.oauth2 import service_account  # type: ignore


class StreamingPullHandler:
    def __init__(self, subscription_path, callback):
        self._subscription_path = subscription_path
        self._callback = callback

    def pull(self):
        service_account_json = settings.GOOGLE_SERVICE_ACCOUNT_JSON
        service_account_info = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info
        )
        subscriber = pubsub_v1.SubscriberClient(credentials=credentials)

        # Limit the subscriber to only have ten outstanding messages at a time.
        flow_control = pubsub_v1.types.FlowControl(max_messages=10)
        streaming_pull_future = subscriber.subscribe(
            self._subscription_path,
            callback=self._callback,
            flow_control=flow_control,
            await_callbacks_on_shutdown=True,
        )

        with subscriber:
            try:
                streaming_pull_future.result()
            except Exception as ex:
                streaming_pull_future.cancel()
                raise ex
