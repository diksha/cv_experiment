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
import json
from typing import Optional

import jwt
import requests
from django.conf import settings
from django.contrib.auth import login as django_login
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth.models import User
from django.core.cache import cache
from django.http import HttpRequest
from loguru import logger

from core.portal.accounts.commands import GetAndMigrateReviewerAccount
from core.portal.accounts.constants import (
    AUTH0_OIDC_EMAIL_CLAIM_KEY,
    AUTH0_UBER_SSO_USER_ID_PREFIX,
)


class Auth0Backend(BaseBackend):
    def _get_public_key(self, jwt_header: dict):
        def fetch_jwks() -> dict:
            if settings.AUTH0_JWKS_URI:
                return requests.get(settings.AUTH0_JWKS_URI).json()
            return {}

        jwks = cache.get_or_set("AUTH0_JWKS_JSON", fetch_jwks)

        for jwk in jwks["keys"]:
            if jwk["kid"] == jwt_header["kid"]:
                return jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
        raise Exception("Public key not found")

    def _get_validated_jwt(self, request: HttpRequest) -> Optional[dict]:
        """Get a validated Auth0 JWT from an HTTP request."""
        if request is None:
            return None

        http_header = request.META.get("HTTP_AUTHORIZATION", None)
        if not http_header:
            return None

        auth_type, encoded_jwt = http_header.split()
        if auth_type != "Bearer":
            return None

        jwt_header = jwt.get_unverified_header(encoded_jwt)
        public_key = self._get_public_key(jwt_header)

        return jwt.decode(
            encoded_jwt,
            public_key,
            audience=settings.AUTH0_JWT_AUDIENCE,
            issuer=settings.AUTH0_JWT_ISSUER,
            algorithms=[jwt_header.get("alg")],
        )

    def authenticate(self, request: HttpRequest, **_):
        auth0_id = None
        try:
            validated_jwt = self._get_validated_jwt(request)
            if not validated_jwt or "sub" not in validated_jwt:
                return None

            auth0_id = validated_jwt.get("sub")
            if not auth0_id:
                return None

            # TODO(PRO-1215): remove this once all Uber accounts have been migrated
            if auth0_id.startswith(AUTH0_UBER_SSO_USER_ID_PREFIX):
                # Looks like an Uber account, use the temporary migration command
                email = validated_jwt.get(AUTH0_OIDC_EMAIL_CLAIM_KEY)
                return GetAndMigrateReviewerAccount(auth0_id, email).execute()

            user = User.objects.get(profile__data__auth0_id=auth0_id)

            # Create a Django session for this user if one doesn't already exist
            if not request.user or not request.user.is_authenticated:
                django_login(
                    request,
                    user,
                    backend="django.contrib.auth.backends.ModelBackend",
                )

            return user
        except User.DoesNotExist:
            logger.error(
                f"No matching Django user was found for Auth0 ID: {auth0_id}"
            )
            return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id, is_active=True)
        except User.DoesNotExist:
            logger.error(
                f"No matching Django user was found for user_id: {user_id}"
            )
            return None
