
variable "grafana_url" {
}

variable "grafana_api_key" {
  sensitive = true
}

variable "grafana_irsa_arn" {

}

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


variable "aws_notifications_slack_app_webhook_url" {
}

variable voxel_services_root_key_base64encoded {
  type = string
  sensitive = true
}

variable voxel_services_root_cert_base64encoded {
  type = string
  sensitive = true
}

variable voxel_services_root_password_base64encoded {
  type = string
  sensitive = true
}