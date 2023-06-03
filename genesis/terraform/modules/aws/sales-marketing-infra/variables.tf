variable "target_account_id" {
  type = string
}

variable "sales_marketing_vpc_cidr_root" {
  type = string
}

variable "sales_marketing_RDS_replica_cidr_root" {
  type = string
}

variable "sales_marketing_RDS_replica_region" {
  type = string
}

variable "environment" {
  type = string
}
variable "primary_region" {
  type = string
}

variable "slack_token" {
  type      = string
  sensitive = true
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

variable "db_instance_size" {
  type = string
}

variable "aws_notifications_slack_app_webhook_url" {
  type      = string
  sensitive = true
}