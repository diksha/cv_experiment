variable "account_id" {}
variable "primary_region" {}
variable "environment" {}
variable "vpc_name" {
}

variable "vpc_id" {
}

variable "private_subnets" {
}

variable "security_group_ids" {
}

variable "cluster_name" {
}

variable "cluster_version" {
  default = "1.24"
}

variable "eks_gpu_instance_types" {
  default = ["g4dn.2xlarge"]
}
variable "eks_cpu_instance_types" {
  default = ["m5.2xlarge"]
}

variable "eks_default_max_instance_count" {
  default = 10
}
variable "dev_mode" {
  default = false
}
variable "eks_should_create_standard_node_group" {
  default = true
}
variable "eks_should_create_gpu_node_group" {
  default = false
}
variable "vpc_cidr_root" {
  type        = string
  description = "Give a single number 'X' denoting the 10.X.0.0/16 CIDR that should be used for the VPC"
}

variable "az_count" {
  default = 2
}

variable "eks_extra_node_policies" {
  type    = map(string)
  default = {}
}

variable "eks_k8s_auth_extra_config" {
  default = []
}

variable "eks_k8s_group_role_bindings" {
  default = {}
}


variable "eks_default_disk_size" {
  default = 100
}

variable eks_extra_node_groups {
  default     = {}
}

variable enable_nginx_ingress {
  default = true
}

variable ingress_tcp_services {
  default = {}
}

variable "ingress_ssl_ports" {
  default = "443"
}

variable "ingress_cert_subject_alternative_names" {
  default = []
}

variable "create_alarm" {
  default = false
}

variable "aws_notifications_slack_app_webhook_url" {
  sensitive = true
  default = ""
}