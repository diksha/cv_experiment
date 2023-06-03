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
    }
    argocd = {
      source  = "oboukili/argocd"
      version = "=4.2.0"
    }
  }
}
