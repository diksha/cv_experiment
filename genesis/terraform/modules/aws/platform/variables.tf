variable "target_account_id" {
  type = string
}

variable "google_client_id" {
  sensitive = true
}

variable "google_client_secret" {
  sensitive = true
}

variable "primary_region" {
  type = string
}

variable "root_domain" {
  type = string
}
variable "subdomain" {
  type = string
}

variable "google_groups_sa_json_base64_encoded" {
  sensitive = true
}

variable "slack_token" {
  sensitive = true
}

variable "vpc_cidr_root" {
}

variable "observability_identifier" {
}

variable "smtp_password" {
  type      = string
  sensitive = true
}

variable "smtp_user" {
  type = string
}
variable "smtp_host" {
  type = string
}

variable "smtp_port" {
  type = string
}

variable "cloudflare_tunnel_token" {
  type      = string
  sensitive = true
}

variable "google_application_credentials_vault_server_base64" {
  sensitive = true
}

variable "retool_license_key" {
  type      = string
  sensitive = true
}

variable "snipeit_key" {
  type      = string
  sensitive = true
}

variable "sentry_slack" {
  type = object({
    client_id      = string
    client_secret  = string
    signing_secret = string
  })
  sensitive = true
}

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