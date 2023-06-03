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
from django.apps import AppConfig


class Config(AppConfig):
    name = "core.portal.api"

    def ready(self):
        # https://docs.djangoproject.com/en/3.2/topics/signals/#connecting-receiver-functions
        # pylint: disable=unused-import
        # pylint: disable=import-outside-toplevel
        import core.portal.api.signals.handlers  # noqa pylint: disable=unused-import
