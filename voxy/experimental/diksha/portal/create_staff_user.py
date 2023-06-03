# trunk-ignore-all(pylint/C0413,flake8/E402)
import argparse

import django

django.setup()

# trunk-ignore(pylint/W0611,flake8/F401): Example usage is commented out
from core.portal.accounts.tools.create_staff_user import StaffUser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_key", "-o", type=str, required=True)
    parser.add_argument("--zone_key", "-z", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    users_to_create = [
        # Example user definition
        # StaffUser(
        #     email="foo@example.com",
        #     password=None,
        #     given_name="Jane",
        #     family_name="Doe",
        #     org_key=args.org_key,
        #     zone_key=args.zone_key,
        #     manager=True,
        #     admin=False
        # ),
    ]
    for user in users_to_create:
        user.create()
