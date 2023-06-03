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
import io
import json
import os
from base64 import b64encode

from auth0.v3.authentication import GetToken
from auth0.v3.management import Auth0
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


User = get_user_model()


class Command(BaseCommand):
    """Imports users from Django username/password auth into Auth0.

    Usage:

        $ AUTH0_DOMAIN=abcd \
            AUTH0_KEY=abcd \
            AUTH0_SECRET=abcd \
            AUTH0_CONNECTION_ID=abcd \
            bazel run core/portal:manage -- migrate
    """

    help = "Take all users in database and import them to Auth0"

    def handle(self, *args, **kwargs) -> None:
        del args, kwargs

        users = []
        for user in User.objects.all():
            if not user.email:
                print(f"User {user.id} has no email.")
                continue

            if not user.password:
                print(f"User {user.id} has no password.")
                continue

            # Format password hash as pbkdf2 hash value
            password_parts = user.password.split("$")
            password_parts[0] = password_parts[0].replace("_", "-")
            password_parts[2] = b64encode(password_parts[2].encode()).decode().replace("=", "")
            password_parts[3] = password_parts[3].replace("=", "")
            password_hash = f"${password_parts[0]}$i={password_parts[1]},l=32${password_parts[2]}${password_parts[3]}"

            user_dict = {
                "email": user.email,
                "email_verified": True,
                "custom_password_hash": {
                    "algorithm": "pbkdf2",
                    "hash": {
                        "value": password_hash,
                        "encoding": "utf8"
                    }
                }
            }

            users.append(user_dict)

        # Uncomment the following lines to preview affected users
        # from pprint import pprint
        # pprint(users)
        # return

        # Dump user list to in-memory file
        json_file = io.StringIO()
        json.dump(users, json_file)
        json_file.seek(0)

        # Collect environment variables
        AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
        AUTH0_KEY = os.getenv("AUTH0_KEY")
        AUTH0_SECRET = os.getenv("AUTH0_SECRET")
        AUTH0_CONNECTION_ID = os.getenv("AUTH0_CONNECTION_ID")

        print("Importing users into Auth0")

        get_token = GetToken(AUTH0_DOMAIN)
        token = get_token.client_credentials(
            AUTH0_KEY,
            AUTH0_SECRET,
            f"https://{AUTH0_DOMAIN}/api/v2/"
        )
        mgmt_api_token = token["access_token"]
        auth0 = Auth0(AUTH0_DOMAIN, mgmt_api_token)
        response = auth0.jobs.import_users(AUTH0_CONNECTION_ID, json_file)
        self.stdout.write(str(response))
        json_file.close()
