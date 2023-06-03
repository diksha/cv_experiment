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
import mock
import pytest
from django.contrib.auth.models import AnonymousUser
from django.http import HttpRequest

from core.portal.accounts.backends import Auth0Backend
from core.portal.accounts.constants import AUTH0_UBER_SSO_USER_ID_PREFIX
from core.portal.testing.factories import UserFactory


@pytest.mark.django_db
@mock.patch.object(Auth0Backend, "_get_validated_jwt")
@mock.patch("core.portal.accounts.backends.django_login")
def test_auth0_authentication_also_login_django(
    mock_login: mock.Mock,
    mock_get_validated_jwt: mock.Mock,
) -> None:
    """Verifies Auth0 backend creates a Django session.

    Args:
        mock_login (mock.Mock): mock Django login function
        mock_get_validated_jwt (mock.Mock): mock JWT validation function
    """

    test_id = "test_id"
    mock_get_validated_jwt.return_value = {"sub": test_id}
    UserFactory(
        profile__data={"auth0_id": test_id},
    )

    request = HttpRequest()
    # anonymous because the user hasn't logged in yet
    request.user = AnonymousUser()

    Auth0Backend().authenticate(request)
    mock_login.assert_called_once()


@pytest.mark.django_db
@mock.patch.object(Auth0Backend, "_get_validated_jwt")
@mock.patch("core.portal.accounts.backends.django_login")
def test_auth0_authentication_does_not_login_already_authed(
    mock_login: mock.Mock,
    mock_get_validated_jwt: mock.Mock,
) -> None:
    """Verifies Auth0 backend does not create Django session if already exists.

    Args:
        mock_login (Callable): mock Django login function
        mock_get_validated_jwt (Callable): mock JWT validation function
    """
    test_id = "test_id"
    mock_get_validated_jwt.return_value = {"sub": test_id}

    user = UserFactory(
        profile__data={"auth0_id": test_id},
    )
    request = HttpRequest()
    request.user = user

    Auth0Backend().authenticate(request)

    mock_login.assert_not_called()


@pytest.mark.django_db
@mock.patch.object(Auth0Backend, "_get_validated_jwt")
@mock.patch(
    # trunk-ignore(pylint/C0301): long line is fine
    "core.portal.accounts.commands.get_and_migrate_reviewer_account.GetAndMigrateReviewerAccount.execute"
)
def test_auth0_authentication_calls_migration_command_for_uber_users(
    mock_execute: mock.Mock,
    mock_get_validated_jwt: mock.Mock,
) -> None:
    """Verifies Auth0 backend calls migration command for Uber users.

    Args:
        mock_execute (Callable): mock migration command function
        mock_get_validated_jwt (Callable): mock JWT validation function
    """
    test_id = f"{AUTH0_UBER_SSO_USER_ID_PREFIX}|test_id"
    mock_get_validated_jwt.return_value = {"sub": test_id}

    user = UserFactory(
        profile__data={"auth0_id": test_id},
    )
    request = HttpRequest()
    request.user = user

    Auth0Backend().authenticate(request)

    mock_execute.assert_called_once()


@pytest.mark.django_db
@mock.patch.object(Auth0Backend, "_get_validated_jwt")
@mock.patch(
    # trunk-ignore(pylint/C0301): long line is fine
    "core.portal.accounts.commands.get_and_migrate_reviewer_account.GetAndMigrateReviewerAccount.execute"
)
def test_auth0_authentication_does_not_call_migration_command_for_non_uber_users(
    mock_execute: mock.Mock,
    mock_get_validated_jwt: mock.Mock,
) -> None:
    """Verifies Auth0 backend does NOT call migration command for non-Uber users.

    Args:
        mock_execute (Callable): mock migration command function
        mock_get_validated_jwt (Callable): mock JWT validation function
    """
    test_id = "test_id"
    mock_get_validated_jwt.return_value = {"sub": test_id}

    user = UserFactory(
        profile__data={"auth0_id": test_id},
    )
    request = HttpRequest()
    request.user = user

    Auth0Backend().authenticate(request)

    mock_execute.not_called()
