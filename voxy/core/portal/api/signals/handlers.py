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
from django.conf import settings
from django.contrib.auth.models import User
from django.db.models.signals import (
    post_delete,
    post_save,
    pre_delete,
    pre_save,
)
from django.dispatch import receiver

from core.portal.accounts.services import initialize_account
from core.portal.api.models.comment import Comment
from core.portal.api.models.incident import Incident, UserIncident
from core.utils.aws_utils import batch_delete_s3_files


@receiver(post_save, sender=User, dispatch_uid="create_user_profile")
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        initialize_account(user_id=instance.pk)


@receiver(post_save, sender=User, dispatch_uid="save_user_profile")
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()


@receiver(pre_save, sender=Incident, dispatch_uid="incident_pre_save")
def incident_pre_save(sender, instance, *args, **kwargs):
    if sender != Incident:
        return

    if not instance.data:
        return

    return


@receiver(post_delete, sender=Incident, dispatch_uid="incident_post_delete")
def incident_post_delete(sender, instance, *args, **kwargs):
    """Cleans up external data after an incident is deleted.

    Ignoring 404 responses for S3 objects because if the path as-stored
    does not match an object, there's not much we can do to recover it.
    Just assume it was already deleted.

    Args:
        sender (Class): Class which triggered the signal.
        instance (object): Object instance which triggered the signal.
        args (list): unused positional args.
        kwargs (dict): unused keyword args.
    """
    if settings.DEVELOPMENT or settings.TEST:
        return

    if sender != Incident:
        return

    if not instance.data:
        return

    batch_delete_s3_files(
        *[
            instance.data.get("video_s3_path"),
            instance.data.get("annotations_s3_path"),
            instance.data.get("video_thumbnail_s3_path"),
        ],
        ignore_404=True,
    )


@receiver(
    post_save, sender=UserIncident, dispatch_uid="assign_activity_record"
)
def assign_activity_record(sender, instance, *args, **kwargs):
    """Create an activity records everytime we assign/unassign user to incident"""
    try:
        assigned_by = User.objects.get(id=instance.assigned_by_id)
        assignee = User.objects.get(id=instance.assignee_id)
    except User.DoesNotExist:
        return

    def get_fullname(user):
        return f"{user.first_name} {user.last_name}"

    obj = Comment(
        text=f"{get_fullname(assigned_by)} assigned to {get_fullname(assignee)}",
        owner_id=instance.assigned_by_id,
        incident_id=instance.incident_id,
        activity_type=Comment.ActivityType.ASSIGN,
        note=instance.note,
    )
    obj.save()


@receiver(
    pre_delete, sender=UserIncident, dispatch_uid="unassign_activity_record"
)
def unassign_activity_record(sender, instance, *args, **kwargs):
    """Create an activity records everytime we assign/unassign user to incident"""
    try:
        assigned_by = User.objects.get(id=instance.assigned_by_id)
        assignee = User.objects.get(id=instance.assignee_id)
    except User.DoesNotExist:
        return

    def get_fullname(user):
        return f"{user.first_name} {user.last_name}"

    obj = Comment(
        text=f"{get_fullname(assignee)} unassigned by {get_fullname(assigned_by)}",
        owner_id=instance.assigned_by_id,
        incident_id=instance.incident_id,
        activity_type=Comment.ActivityType.ASSIGN,
    )
    obj.save()
