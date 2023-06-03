variable account_id {

}
variable "cluster_name" {
  type = string
}

variable "google_application_credentials_base64_encoded" {
  sensitive = true
}

variable sentry_dsn {
  sensitive = true
}

variable "environment" {
  
}

variable "oidc_provider" {
  
}

variable "region" {
  type = string
}