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
import bleach
from django.contrib.auth.models import User
from django.db import models

from core.portal.lib.models.base import Model


class Comment(Model):
    class ActivityType(models.TextChoices):
        COMMENT = ("comment", "Comment")
        ASSIGN = ("assign", "Assign")
        LOG = ("log", "Log")
        RESOLVE = ("resolve", "Resolve")
        REOPEN = ("reopen", "Reopen")

    text = models.CharField(max_length=1000, null=False, blank=False)
    owner = models.ForeignKey(
        User, blank=True, null=True, on_delete=models.SET_NULL
    )
    incident = models.ForeignKey(
        "Incident",
        blank=True,
        null=True,
        on_delete=models.SET_NULL,
        related_name="comments",
    )
    activity_type = models.CharField(
        max_length=100, null=True, blank=True, choices=ActivityType.choices
    )
    note = models.CharField(max_length=1000, null=True, blank=True)

    def __str__(self):
        text_preview = self.text[:50] + (self.text[:50] and "...")
        return f"Incident #{self.incident.id}: {text_preview}"

    def save(self, *args, **kwargs):
        # Sanitize user input
        self.text = bleach.clean(self.text)
        super().save(*args, **kwargs)
