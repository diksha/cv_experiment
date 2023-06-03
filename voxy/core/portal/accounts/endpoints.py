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
from auth0.v3.exceptions import Auth0Error
from django.utils import timezone
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.exceptions import ValidationError
from rest_framework.request import Request
from rest_framework.response import Response

from core.portal.accounts.clients.auth0 import Auth0ManagementClient
from core.portal.accounts.services import register_user
from core.portal.api.models.invitation import Invitation


@api_view(["GET", "POST"])
@permission_classes([permissions.AllowAny])
def register(request: Request, **kwargs: str) -> Response:
    """User registration handlers.

    :param request: request sent to register invitation
    :param kwargs: key word arguments
    :raises RuntimeError: if failed validation checks
    :returns: Response.
    """

    invitation = Invitation.objects.filter(
        token=kwargs["token"], expires_at__gte=timezone.now()
    ).get()

    invitee = invitation.invitee
    invited_by = invitation.invited_by

    if request.method == "GET":
        invited_by_name = (
            f"{invited_by.first_name} {invited_by.last_name}".strip()
        )
        return Response(
            dict(
                organization_name=invitation.organization.name,
                email=invitee.email,
                invited_by_name=invited_by_name,
            )
        )

    if request.method == "POST":
        missing_fields = []

        first_name = request.data.get("first_name")
        if not first_name:
            missing_fields.append("first_name")

        last_name = request.data.get("last_name")
        if not last_name:
            missing_fields.append("last_name")

        password = request.data.get("password")
        if not password:
            missing_fields.append("password")

        if missing_fields:
            missing_fields_list = ", ".join(missing_fields)
            raise ValidationError(
                f"The following fields are required: {missing_fields_list}"
            )

        try:
            auth0_client = Auth0ManagementClient()
            register_user(
                auth0_client, invitation, first_name, last_name, password
            )
            return Response(dict(success=True))
        except Auth0Error as auth0_error:
            if "PasswordStrengthError" in auth0_error.message:
                return Response(
                    dict(
                        success=False,
                        error_message=(
                            "Your password is too weak. Make sure it meets"
                            " the password requirements and isn't something"
                            " easy to guess, such as Password123."
                        ),
                    )
                )
            raise auth0_error

    return Response(dict(success=False))
