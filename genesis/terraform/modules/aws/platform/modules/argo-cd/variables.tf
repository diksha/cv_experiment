variable "cluster_name" {
  type = string
}

variable "account_id" {}

variable "google_client_id" {
  type = string
}

variable "google_client_secret" {
  type      = string
  sensitive = true
}

variable "domain" {
  type = string
}


variable "google_groups_sa_json_base64_encoded" {
  sensitive = true
}

variable "argo_slack_token" {
  sensitive = true
}