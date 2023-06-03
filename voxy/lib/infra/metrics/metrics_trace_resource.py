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

import os

from opentelemetry.sdk.resources import Resource

from lib.infra.metrics.constants import CommonConstants


class MetricsTraceResource:
    """
    This class is used to create a resource for the metrics and tracing
    Args:
        service_name (str): Name of the service
        attributes (dict): attribute to be added to the metrics library
    """

    def __init__(self, service_name: str, attributes: dict) -> None:
        default_attributes = {
            "service.name": service_name,
            "image_tag": os.getenv(
                CommonConstants.IMAGE_TAG, CommonConstants.UNKNOWN
            ),
        }
        merged_attributes = default_attributes | attributes
        self._resource = Resource(attributes=merged_attributes)

    def get_metrics_trace_resource(self) -> Resource:
        """
        Get the resource used to create spans and metrics

        Returns:
            Resource: The resource used to create spans and metrics
        """
        return self._resource
