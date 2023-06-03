provider "aws" {
  region = var.primary_region
  assume_role {
    role_arn = "arn:aws:iam::${var.target_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}
