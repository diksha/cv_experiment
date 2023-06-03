terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=4.24.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~>2.16.1"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~>2.6.0"
    }
    grafana = {
      source  = "grafana/grafana"
      version = "~>1.24.0"
    }
  }
}
