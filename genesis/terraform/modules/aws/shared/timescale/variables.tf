variable "target_account_id" {
  type = string
}

variable "primary_region" {
  type = string
}

variable "cluster_name" {
  type = string
}

variable "db_identifier" {
  type = string
}

variable "wait_for_db" {
  type = bool
  default = true
}

variable "chart_version"{
  default = "9.2.7"
}