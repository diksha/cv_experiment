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
import graphene
from django.shortcuts import get_object_or_404

from core.portal.accounts.permissions import INCIDENTS_COMMENT
from core.portal.accounts.permissions_manager import has_zone_permission
from core.portal.api.models.comment import Comment
from core.portal.api.models.incident import Incident
from core.portal.comments.graphql.types import CommentType
from core.portal.lib.graphql.exceptions import PermissionDenied
from core.portal.lib.graphql.mutations import BaseMutation
from core.portal.lib.graphql.utils import pk_from_global_id


class CreateComment(BaseMutation):
    class Arguments:
        incident_id = graphene.ID(required=True)
        text = graphene.String(required=True)

    comment = graphene.Field(CommentType)

    @staticmethod
    def mutate(
        root: None, info: graphene.ResolveInfo, incident_id: str, text: str
    ) -> "CreateComment":
        _, pk = pk_from_global_id(incident_id)
        incident = get_object_or_404(Incident, pk=pk)

        if not has_zone_permission(
            info.context.user, incident.zone, INCIDENTS_COMMENT
        ):
            raise PermissionDenied(
                "You do not have permission to comment on this incident."
            )
        comment_instance = Comment(
            text=text,
            owner_id=info.context.user.id,
            incident_id=pk,
            activity_type="comment",
        )
        comment_instance.save()
        return CreateComment(comment=comment_instance)
