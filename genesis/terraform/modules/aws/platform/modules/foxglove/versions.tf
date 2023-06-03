terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=4.4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">=2.11.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">=2.6.0"
    }
  }
}