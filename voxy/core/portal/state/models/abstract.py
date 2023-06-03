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


from django.db import models
from django_cte import CTEManager
from django_cte.cte import CTEQuerySet
from timescale.db.models.managers import TimescaleManager
from timescale.db.models.querysets import TimescaleQuerySet


class CustomQuerySet(CTEQuerySet, TimescaleQuerySet):
    pass


class CustomManager(TimescaleManager, CTEManager):
    def get_queryset(self):
        return CustomQuerySet(self.model, using=self._db)


class AbstractModel(models.Model):
    class Meta:
        app_label = "state"
        abstract = True

    objects = CustomManager()
