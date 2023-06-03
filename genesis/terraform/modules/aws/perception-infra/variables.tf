variable "target_account_id" {
  type = string
}
variable "google_application_credentials_base64_encoded" {
  type      = string
  sensitive = true
}

variable "perception_vpc_cidr_root" {
  type = string
}

variable "environment" {
  type = string
}
variable "primary_region" {
  type = string
}

variable "slack_token" {
  type = string
  sensitive = true
}
variable "new_relic_license_key" {
  type = string
  sensitive = true
}

variable "observability_identifier" {
  type = string
}

variable "grafana_url" {
  type = string
}

variable "grafana_api_key" {
  type      = string
  sensitive = true
}

variable "grafana_irsa_arn" {
  type = string
}

variable "eks_default_max_instance_count" {
  default = 10
}

variable "perception_eks_gpu_instance_types_comma_separated" {
  default = "g4dn.xlarge"
}

variable "perception_eks_cpu_instance_types_comma_separated" {
  default = "m5.xlarge"
}

variable "aws_notifications_slack_app_webhook_url" {
  type      = string
  sensitive = true
}