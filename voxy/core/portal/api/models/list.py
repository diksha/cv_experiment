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
from django.contrib.auth.models import User
from django.db import models

from core.portal.api.models.incident import Incident
from core.portal.lib.models.base import Model

STARRED_LIST_NAME = "Starred"


class List(Model):

    # TODO: design a permissions/access control solution for sharing lists

    name = models.CharField(max_length=250, null=False, blank=False)
    # NOTE: starred == bookmarked
    is_starred_list = models.BooleanField(
        max_length=250, null=False, blank=False
    )
    owner = models.OneToOneField(User, on_delete=models.CASCADE)
    incidents = models.ManyToManyField(
        Incident, related_name="lists", blank=True
    )

    def __str__(self):
        return f"{self.name} (owned by {self.owner.email})"
