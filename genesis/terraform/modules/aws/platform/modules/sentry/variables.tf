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

variable "slack" {
  type = object({
    client_id      = string
    client_secret  = string
    signing_secret = string
  })
  sensitive = true
}