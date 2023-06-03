variable "cluster_name" {
  type = string
}
variable "account_id" {}


variable "primary_region" {
  type    = string
  default = "us-west-2"
}

variable "google_application_credentials_vault_server_base64" {
  sensitive = true
}

variable "oidc_provider" {}
