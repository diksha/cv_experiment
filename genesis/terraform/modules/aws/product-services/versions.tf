terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=4.4.0"
    }
    kubernetes = {
      source = "hashicorp/kubernetes"
    }
    helm = {
      source = "hashicorp/helm"
      version    = ">=2.7.1"

    }
    argocd = {
      source  = "oboukili/argocd"
      version = "=4.2.0"
    }
  }
}
