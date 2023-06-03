terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=4.24.0"
    }

    local = {
      version = ">=2.4.0"
    }
  }
}
