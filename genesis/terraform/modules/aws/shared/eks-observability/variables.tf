variable "cluster_name" {
  type = string
}

variable "account_id" {}

variable "aws_region" {}

variable "force_deploy_dashboard_config_maps" {
  default = false
}

variable "register_with_grafana" {
  default = false
}

variable "grafana_url" {
  default = "http://localhost:8988"
}

variable "grafana_api_key" {
  sensitive = true
}

variable "observability_identifier" {
}

variable "grafana_irsa_arn" {
}

variable "collector_debug_mode" {
  default = false
}
variable "install_gpu_components" {
  default = false
}

variable "prometheus_memory" {
  default = "4G"
}