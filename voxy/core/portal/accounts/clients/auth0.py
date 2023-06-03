from auth0.v3.authentication import GetToken
from auth0.v3.management import Auth0
from django.conf import settings


class Auth0ManagementClient(Auth0):
    """Auth0 management API client."""

    def __init__(self):
        """Initializes the underlying Auth0 client.

        NOTE: when making management API requests we need to use the
              Auth0 tenant domain, not a custom domain. So in production
              this means we use "voxelprod.us.auth0.com" instead of
              "auth.voxelai.com".

              Auth0 docs:
              https://auth0.com/docs/customize/custom-domains/configure-features-to-use-custom-domains#apis
        """
        domain = settings.AUTH0_TENANT_DOMAIN
        management_client_id = settings.AUTH0_MANAGEMENT_CLIENT_ID
        management_client_secret = settings.AUTH0_MANAGEMENT_CLIENT_SECRET

        get_token = GetToken(domain)
        token = get_token.client_credentials(
            management_client_id,
            management_client_secret,
            f"https://{domain}/api/v2/",
        )
        mgmt_api_token = token["access_token"]

        super().__init__(domain, mgmt_api_token)
