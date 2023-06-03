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
from typing import Optional

import graphene

from core.portal.accounts.graphql.types import UserType
from core.portal.organizations.graphql.types import OrganizationType
from core.portal.session.graphql.mutations import (
    CurrentUserAddBookmark,
    CurrentUserOrganizationUpdate,
    CurrentUserRemoveBookmark,
    CurrentUserSiteUpdate,
)


class SessionQueries(graphene.ObjectType):
    current_user = graphene.Field(UserType)
    current_organization = graphene.Field(OrganizationType)

    def resolve_current_user(self, info: graphene.ResolveInfo) -> UserType:
        return info.context.user

    def resolve_current_organization(
        self, info: graphene.ResolveInfo
    ) -> Optional[OrganizationType]:
        current_user = info.context.user
        if current_user:
            return current_user.profile.current_organization
        return None


class SessionMutations(graphene.ObjectType):
    current_user_add_bookmark = CurrentUserAddBookmark.Field()
    current_user_remove_bookmark = CurrentUserRemoveBookmark.Field()
    current_user_organization_update = CurrentUserOrganizationUpdate.Field()
    current_user_site_update = CurrentUserSiteUpdate.Field()
