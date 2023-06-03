variable "target_account_id" {
  type = string
}

variable "product_vpc_cidr_root" {
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

variable "db_instance_size" {
  type = string
}

variable "redis_instance_type" {
  type = string
}

variable "redis_cluster_size" {
  type = number
}

variable "portal_RDS_replica_cidr_root" {
  type = string
}

variable "portal_RDS_replica_region" {
  type = string
}

variable "portal_eks_cpu_instance_types_comma_separated" {
  default = "m5.xlarge"
}

variable "performance_insights_retention_period" {
  default = 7
}

variable "aws_notifications_slack_app_webhook_url" {
  type      = string
  sensitive = true
}

variable services_root_ca {
  type = object({
    cert_pem = string
    key_pem = string
    password = string
  })
  sensitive = true
}
