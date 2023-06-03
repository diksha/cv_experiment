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

from core.portal.accounts.graphql.schema import UserMutations
from core.portal.analytics.graphql.schema import AnalyticsQueries
from core.portal.comments.graphql.schema import (
    CommentMutations,
    CommentQueries,
)
from core.portal.compliance.graphql.schema import ProductionLineQueries
from core.portal.devices.graphql.schema import (
    CameraConfigNewMutations,
    CameraMutations,
    CameraQueries,
)
from core.portal.incident_feedback.graphql.schema import (
    IncidentFeedbackMutations,
    IncidentFeedbackQueries,
)
from core.portal.incidents.graphql.schema import (
    IncidentMutations,
    IncidentQueries,
    IncidentTypeQueries,
)
from core.portal.integrations.graphql.schema import IntegrationsQueries
from core.portal.lib.graphql.deprecated import DeprecatedQueries
from core.portal.organizations.graphql.schema import (
    OrganizationMutations,
    OrganizationQueries,
)
from core.portal.session.graphql.schema import SessionMutations, SessionQueries
from core.portal.zones.graphql.schema import ZoneMutations, ZoneQueries

# Note: graphene.ObjectType should not be before the custom class
# TypeError: Cannot create a consistent method resolution order (MRO) for bases object


class Mutation(
    CameraConfigNewMutations,
    CameraMutations,
    CommentMutations,
    IncidentFeedbackMutations,
    IncidentMutations,
    OrganizationMutations,
    SessionMutations,
    UserMutations,
    ZoneMutations,
    graphene.ObjectType,
):
    pass


class Query(
    AnalyticsQueries,
    CameraQueries,
    CommentQueries,
    DeprecatedQueries,
    IncidentFeedbackQueries,
    IncidentQueries,
    IncidentTypeQueries,
    IntegrationsQueries,
    OrganizationQueries,
    SessionQueries,
    ZoneQueries,
    ProductionLineQueries,
    graphene.ObjectType,
):
    pass


schema = graphene.Schema(query=Query, mutation=Mutation)
