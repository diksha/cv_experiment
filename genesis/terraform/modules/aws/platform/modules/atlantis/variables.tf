variable "cluster_name" {
  type = string
}

variable "account_id" {}

variable "google_sa_json_terraform_base64_encoded" {
  type = string
  sensitive = true
}

variable "atlantis" {
  type = object({
    github_app_pem      = string
    github_app_secret   = string
  })
  sensitive = true
}

variable "oidc_provider" {}
