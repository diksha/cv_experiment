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
from datetime import timedelta, timezone
from typing import Any

import graphene


class ClientContext:
    def __init__(self, root_context: graphene.Context):
        self.user_timezone_offset = int(
            root_context.META.get("HTTP_X_VOXEL_CLIENT_TIMEZONE_OFFSET", 0)
        )

    @property
    def timezone(self) -> timezone:
        # IMPORTANT: offset must be inverted (-) to produce correct timedleta
        return timezone(timedelta(minutes=-self.user_timezone_offset))


class LocalizationMiddleware:
    """Attaches localization context to the root context."""

    def resolve(
        self,
        next_middleware: Any,
        root: Any,
        info: graphene.ResolveInfo,
        **args: Any
    ) -> Any:
        info.context.client = ClientContext(info.context)
        return next_middleware(root, info, **args)
