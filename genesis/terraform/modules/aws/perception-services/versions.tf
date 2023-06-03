terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=4.24.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~>2.12.1"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~>2.6.0"
    }
    argocd = {
      source  = "oboukili/argocd"
      version = "=4.2.0"
    }
  }
}
