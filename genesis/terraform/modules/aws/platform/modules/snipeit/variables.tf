variable "cluster_name" {
  type = string
}

variable "account_id" {}

variable "snipeit_key" {
  type      = string
  sensitive = true
}