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
from graphene_django import DjangoObjectType

from core.portal.api.models.comment import Comment
from core.portal.incidents.graphql.types import IncidentType


class CommentType(DjangoObjectType):
    class Meta:
        model = Comment
        interfaces = [graphene.relay.Node]
        fields = "__all__"

    incident = graphene.Field(IncidentType)

    @staticmethod
    def resolve_incident(
        parent: Comment, _: graphene.ResolveInfo
    ) -> IncidentType:
        return parent.incident
