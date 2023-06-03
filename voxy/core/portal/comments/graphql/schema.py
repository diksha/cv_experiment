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
from typing import List

import graphene
from django.db.models.query import QuerySet

from core.portal.api.models.comment import Comment
from core.portal.comments.graphql.mutations import CreateComment
from core.portal.comments.graphql.types import CommentType
from core.portal.lib.graphql.utils import pk_from_global_id


class CommentQueries(graphene.ObjectType):

    comments = graphene.List(
        CommentType, incident_id=graphene.ID(required=True)
    )
    recent_comments = graphene.List(CommentType)

    def resolve_comments(
        self, info: graphene.ResolveInfo, incident_id: str
    ) -> QuerySet[Comment]:
        _, incident_pk = pk_from_global_id(incident_id)
        return Comment.objects.filter(
            incident__organization=info.context.user.profile.current_organization,
            incident__pk=incident_pk,
        )

    def resolve_recent_comments(
        self, info: graphene.ResolveInfo
    ) -> List[Comment]:
        return Comment.objects.filter(
            incident__organization=info.context.user.profile.current_organization,
            incident__camera__zone_id__in=[info.context.user.profile.site],
        ).order_by("-created_at")[:15]


class CommentMutations(graphene.ObjectType):

    create_comment = CreateComment.Field()
