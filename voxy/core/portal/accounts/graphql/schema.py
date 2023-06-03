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

from core.portal.accounts.graphql.mutations import (
    ReviewerAccountCreate,
    ReviewerAccountRoleUpdate,
    UserInvite,
    UserMFAUpdate,
    UserNameUpdate,
    UserRemove,
    UserResendInvitation,
    UserRoleUpdate,
    UserUpdate,
    UserZonesUpdate,
)


class UserMutations(graphene.ObjectType):
    user_invite = UserInvite.Field()
    user_update = UserUpdate.Field()
    user_zones_update = UserZonesUpdate.Field()
    user_role_update = UserRoleUpdate.Field()
    user_remove = UserRemove.Field()
    user_resend_invitation = UserResendInvitation.Field()
    reviewer_account_create = ReviewerAccountCreate.Field()
    reviewer_account_role_update = ReviewerAccountRoleUpdate.Field()
    user_mfa_update = UserMFAUpdate.Field()
    user_name_update = UserNameUpdate.Field()
