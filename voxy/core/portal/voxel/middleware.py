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
import os
from typing import Callable

from django.http import HttpRequest, HttpResponse


class CustomHeaderMiddleware:
    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)
        response["X-API-Revision"] = os.getenv("GIT_REVISION") or "unknown"
        return response


# In addition to Django's default, we want to provide these additional headers
class CustomSecurityMiddleware:
    def __init__(self, get_response: Callable) -> None:
        self.get_response = get_response

    def _enforce_content_security_policy(self, request: HttpRequest) -> bool:
        """Determine if content security policy should be enforced for this requeset.

        Args:
            request (HttpRequest): HTTP request

        Returns:
            bool: true if CSP should be enforced, otherwise false
        """
        # Don't enforce CSP for Graphiql view, it loads scripts from a CDN.
        # Actual API requests will always be POST requests, only the Graphiql
        # view uses GET requests.
        if request.path == "/graphql/" and request.method == "GET":
            return False
        if request.path.startswith("/internal/backend/django_plotly_dash/"):
            return False
        return True

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Django call invocation on middleware

        Args:
            request (HttpRequest): request being sent

        Returns:
            response (Callable): response with added headers
        """
        response = self.get_response(request)
        response[
            "Access-Control-Allow-Headers"
        ] = "X-Requested-With,content-type"
        response["Access-Control-Allow-Credentials"] = "true"
        response["Access-Control-Max-Age"] = "86400"
        response["Access-Control-Expose-Headers"] = "Content-Length"
        response["X-Content-Type-Options"] = "nosniff"
        response["X-XSS-Protection"] = "1; mode=block"
        response["X-Permitted-Cross-Domain-Policies"] = "none"

        if self._enforce_content_security_policy(request):
            response[
                "Content-Security-Policy"
            ] = "script-src 'self' 'unsafe-inline' 'unsafe-eval'"

        # remove server header/date to prevent server version leaks
        del response["Server"]
        del response["Date"]
        return response
