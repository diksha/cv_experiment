variable "cluster_name" {
  type = string
}

variable "account_id" {}

variable "static_ipv4s" {
  type = string
}

variable "subnet_ids" {
  type = string
}

variable "enable_nginx_ingress" {
  default = true
}

variable "tcp_services" {
  default = {}
}

variable "ssl_ports" {
  default = "443"
}

variable "subject_alternative_names" {
  default = []
}