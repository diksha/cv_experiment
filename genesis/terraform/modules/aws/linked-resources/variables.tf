variable "root_account_id" {
}
variable "platform_account_id" {
}
variable "production_account_id" {
}
variable "staging_account_id" {
}
variable "development_account_id" {
}

variable "prime_account_id" {
}

variable "galileo_account_id" {
}

variable "primary_region" {}

variable "slack_token" {
  sensitive = true
}

variable "argo_admin_password" {
  sensitive = true
}

variable "argo_admin_username" {}

variable "argo_domain" {}

variable "portal_production_postgres_pwd" {
  sensitive = true
}
variable "portal_production_timescale_pwd" {
  sensitive = true
}

variable "portal_staging_postgres_pwd" {
  sensitive = true
}

variable "postgres_perception_scenario_eval_password" {
  sensitive = true
}

variable "sales_marketing_production_postgres_pwd" {
  sensitive = true
}

variable "grafana_url" {
}
variable "grafana_api_key" {
  sensitive = true
}

variable "grafana_irsa_arn" {}
