terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=4.24.0"
    }
    kubernetes = {
      source = "hashicorp/kubernetes"
      version = ">=2.18.0"
    }
    helm = {
      source = "hashicorp/helm"
      version = ">=2.8.0"
    }
    argocd = {
      source  = "oboukili/argocd"
      version = "=4.2.0"
    }
    grafana = {
      source  = "grafana/grafana"
      version = "~>1.29.0"
    }
  }
}
