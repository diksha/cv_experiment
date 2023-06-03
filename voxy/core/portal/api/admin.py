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
from django.contrib import admin

from core.portal.api.models.comment import Comment
from core.portal.api.models.incident import (
    Incident,
    IncidentAdmin,
    UserIncident,
    UserIncidentAdmin,
)
from core.portal.api.models.incident_feedback import (
    IncidentFeedback,
    IncidentFeedbackAdmin,
)
from core.portal.api.models.incident_type import (
    CameraIncidentType,
    IncidentType,
    OrganizationIncidentType,
    SiteIncidentType,
)
from core.portal.api.models.invitation import Invitation
from core.portal.api.models.organization import Organization, OrganizationAdmin
from core.portal.api.models.profile import Profile, ProfileAdmin
from core.portal.api.models.share_link import ShareLink
from core.portal.state.models.state import State

# Register your models here.
admin.site.register(Incident, IncidentAdmin)
admin.site.register(IncidentFeedback, IncidentFeedbackAdmin)
admin.site.register(Profile, ProfileAdmin)
admin.site.register(Invitation)
admin.site.register(Organization, OrganizationAdmin)
admin.site.register(Comment)
admin.site.register(IncidentType)
admin.site.register(OrganizationIncidentType)
admin.site.register(UserIncident, UserIncidentAdmin)
admin.site.register(State)
admin.site.register(ShareLink)
admin.site.register(CameraIncidentType)
admin.site.register(SiteIncidentType)
