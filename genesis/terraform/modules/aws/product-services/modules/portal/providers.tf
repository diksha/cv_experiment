provider "auth0" {
  domain = local.domains[var.context.environment]
  client_id = local.client_id
  client_secret = local.client_secret
}

 provider "aws" {
  region = local.mumbai_region
  alias = "mumbai"
  assume_role {
    role_arn = "arn:aws:iam::${var.context.target_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}
