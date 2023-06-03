variable "target_account_id" {
  type = string
}

variable "google_application_credentials_base64_encoded" {
  type      = string
  sensitive = true
}

variable "environment" {
  type = string
}

variable "primary_region" {
  type = string
}

variable "eks_cluster_name" {
  type = string
}

variable "sentry_dsn" {
  type = string
  sensitive = true
}

variable "argo_server_domain" {}
variable "argo_server_port" {
  default = 443
}
variable "argo_username" {}
variable "argo_password" {
  sensitive = true
}

variable "states_events_bucket_arn" {
  type = string
}

variable "prime_account_id" {
  type = string
}

variable "services_root_ca" {
  type = object({
    cert_pem = string
    key_pem = string
    password = string
  })
  sensitive = true
}