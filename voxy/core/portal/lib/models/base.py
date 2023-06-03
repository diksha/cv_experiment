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
from typing import List, Tuple

import pytz
from django.db import models

# TODO(PORTAL-166): get mypy working with model fields
# trunk-ignore-all(mypy/var-annotated)


class Model(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        abstract = True
        app_label = "api"

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    @staticmethod
    def timezones() -> List[Tuple[str, str]]:
        return list((s, s) for s in pytz.common_timezones)
