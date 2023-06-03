variable "cluster_name" {
  type = string
}

variable "db_identifier" {
  type = string
}

variable "account_id" {
  type = string
}


variable "wait_for_db" {
  type = bool
  default = true
}

variable "cluster_size" {
  default = 1
}
