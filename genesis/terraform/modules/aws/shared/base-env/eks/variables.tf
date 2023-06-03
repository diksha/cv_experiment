variable "account_id" {}

variable "cluster_name" {
  type = string
}

variable "cluster_version" {
  default = "1.21"
}

variable "dev_mode" {
  default = false
}

variable "environment" {
}

variable "subnet_ids" {}
variable "security_group_ids" {}


variable "vpc_id" {}


variable "default_disk_size_gb" {
  type    = number
  default = 100
}

variable "instance_types" {
  default = ["m5.xlarge", "m5.4xlarge"]
}

variable "gpu_instance_types" {
  default = ["g4dn.2xlarge"]
}

variable "cpu_instance_types" {
  default = ["m5.2xlarge"]
}
variable "cpu_instance_min_size" {
  default = 1
}
variable "cpu_instance_max_size" {
  default = 10
}

variable "gpu_instance_min_size" {
  default = 0
}
variable "gpu_instance_max_size" {
  default = 10
}

variable "should_create_standard_node_group" {
  default = true
}

variable "should_create_gpu_node_group" {
  default = false
}

variable "extra_node_policies" {
  type    = map(string)
  default = {}
}

variable "k8s_auth_extra_config" {
  default = []
}

variable "k8s_group_role_bindings" {
  default = {}
}


variable "extra_node_groups" {
  default = {}
}

variable "create_alarm" {
  default = false
}

variable "aws_notifications_slack_app_webhook_url" {
  sensitive = true
}

variable "vpc_name" {
}