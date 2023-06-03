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
from typing import Callable

from django.contrib.auth import authenticate
from django.http import HttpRequest, HttpResponse


class StatelessAuthenticationMiddleware:
    """Authenticates evey request for stateless auth methods, such as JWTs."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = authenticate(request)
        if user is not None and user.is_authenticated:
            request.user = user
        response = self.get_response(request)
        return response


class PermissionsMiddleware:
    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        if request.user is not None and request.user.is_authenticated:
            request.user.permissions = request.user.profile.permissions

        response = self.get_response(request)
        return response
