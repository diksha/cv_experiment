
variable "deplomyment_identifier"{
  default = "sematic"
}

variable "eks_cluster_name" {
  type = string
}
variable "account_id" {}

variable "primary_region" {
  type = string
}

variable "perception_verbose_slack_hook_url_token" {
  sensitive = true
}

variable "oidc_provider" {
  type = string
}

variable "google_client_id" {
  type = string
}

variable "environment" {
  type = string
}

variable "private_subnet_ids" {
}

variable "private_subnet_cidrs" {
}

variable "vpc_id" {
  type = string
}