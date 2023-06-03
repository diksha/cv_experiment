variable "cluster_name" {
  type = string
}

variable "account_id" {}

variable "google_client_id" {
  type = string
}

variable "google_client_secret" {
  type      = string
  sensitive = true
}

variable "retool_license_key" {
  type      = string
  sensitive = true
}

variable "eks_cluster_name" {
  type = string
}