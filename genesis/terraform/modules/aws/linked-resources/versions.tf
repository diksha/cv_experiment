terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">=4.24.0"
    }
    grafana = {
      source  = "grafana/grafana"
      version = "1.29.0"
    }
  }
}
