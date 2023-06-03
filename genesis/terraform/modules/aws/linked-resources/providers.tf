provider "aws" {
  alias  = "root"
  region = var.primary_region
  assume_role {
    role_arn = "arn:aws:iam::${var.root_account_id}:role/TerraformAccess"
  }
}

provider "aws" {
  alias  = "production"
  region = var.primary_region
  assume_role {
    role_arn = "arn:aws:iam::${var.production_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}

provider "aws" {
  alias  = "staging"
  region = var.primary_region
  assume_role {
    role_arn = "arn:aws:iam::${var.staging_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}

provider "aws" {
  alias  = "development"
  region = var.primary_region
  assume_role {
    role_arn = "arn:aws:iam::${var.development_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}

provider "aws" {
  alias  = "prime"
  region = var.primary_region
  assume_role {
    role_arn = "arn:aws:iam::${var.prime_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}

provider "aws" {
  alias  = "platform"
  region = var.primary_region
  assume_role {
    role_arn = "arn:aws:iam::${var.platform_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}

provider "aws" {
  alias  = "galileo"
  region = var.primary_region
  assume_role {
    role_arn = "arn:aws:iam::${var.galileo_account_id}:role/TerraformAccess"
  }
  default_tags {
    tags = {
      map-migrated = "d-server-00swbp99drezfh"
    }
  }
}

provider "grafana" {
  url    = var.grafana_url
  auth   = var.grafana_api_key
  org_id = 1
}

provider "grafana" {
  alias  = "reviewers"
  url    = var.grafana_url
  auth   = var.grafana_api_key
  org_id = 211
}
