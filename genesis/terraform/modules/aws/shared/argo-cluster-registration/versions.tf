terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=4.4.0"
    }
    argocd = {
      source  = "oboukili/argocd"
      version = "=4.2.0"
    }
  }
}