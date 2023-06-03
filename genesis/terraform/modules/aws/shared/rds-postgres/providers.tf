provider "aws" {
  alias  = "replica"
  region = var.replica_region
  assume_role {
    role_arn = "arn:aws:iam::${var.account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}

