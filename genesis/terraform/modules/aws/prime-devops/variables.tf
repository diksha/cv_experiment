variable context {
  type = object({
    accounts = object({
      root = object({id = number})
      prime = object({id = number})
      development = object({id = number})
      staging = object({id = number})
      platform = object({id = number})
      production = object({id = number})
    })
  })
}

variable "target_account_id" {
}

variable "buildkite_token" {
  sensitive = true
}

variable "perception_verbose_slack_hook_url_token" {
  sensitive = true
}

variable "environment" {
}

variable "primary_region" {
}

variable "git_ssh_key_base64_encoded" {
  sensitive = true
}

variable "slack_token" {
  sensitive = true
}

variable "devops_vpc_cidr_root" {
  
}

variable "observability_identifier" {
  
}

variable "grafana_url" {
}

variable "grafana_api_key" {
  sensitive = true
}

variable "grafana_irsa_arn" {

}

variable "google_client_id" {
  sensitive = true
}

variable "google_client_secret" {
  sensitive = true
}

variable "root_account_id" {
}

variable "services_root_ca" {
  type = object({
    cert_pem = string
    key_pem = string
    password = string
  })
  sensitive = true
}