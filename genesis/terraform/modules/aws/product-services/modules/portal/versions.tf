terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=4.4.0"
      configuration_aliases = [ aws, aws.east ]
    }
    kubernetes = {
      source = "hashicorp/kubernetes"
    }
    helm = {
      source = "hashicorp/helm"
    }
    argocd = {
      source  = "oboukili/argocd"
      version = "=4.2.0"
    }
    auth0 = {
      source  = "auth0/auth0"
    }
  }
}
