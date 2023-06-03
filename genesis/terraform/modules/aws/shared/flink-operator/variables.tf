variable "cluster_name" {
  type = string
}

variable "account_id" {}

variable "namespace_to_watch" {
  description = "Namespace to watch for Flink Jobs"
  type = string
}
