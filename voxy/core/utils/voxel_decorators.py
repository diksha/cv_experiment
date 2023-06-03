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

import time
from functools import wraps

from loguru import logger


def retry_handler(
    exceptions,
    max_retry_count=1,
    retry_delay_seconds=0.5,
    backoff_factor=2,
):
    """
    Retry decorator for functions with api calls
    Adapted from https://towardsdatascience.com/are-you-using-python-with-apis-learn-how-to-use-a-retry-decorator-27b6734c3e6

    Args:
        exceptions: Single exception or tuple of exceptions to retry for
        max_retry_count (int): number of retries before function fails
        retry_delay_seconds (float): time in seconds to delay the retry mechanism
        backoff_factor (int): scalar which increases the delay each retry

    Returns:
        Function decorator
    """

    def retry_decorator(func):
        @wraps(func)
        def retry_wrapper(*args, **kwargs):
            remaining_tries = max_retry_count + 1
            retry_delay = retry_delay_seconds
            while remaining_tries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as exception:
                    remaining_tries -= 1
                    if remaining_tries == 0:
                        raise exception
                    logger.warning(
                        f"{func.__name__} encountered {exception}, retrying, remaining retries, {remaining_tries}"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= backoff_factor

        return retry_wrapper

    return retry_decorator
