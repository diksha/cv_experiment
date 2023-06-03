variable "cluster_name" {
  type = string
}

variable "db_identifier" {
  type = string
}

variable "namespace" {
  type = string
  default = ""
}


variable "create_namespace" {
  type = bool
  default = true
}

variable "account_id" {
  type = string
}

variable "init_db_script_config_map" {
  type = string
  default = ""
}

variable "wait_for_db" {
  type = bool
  default = true
}

variable "replica_count" {
  default = 3
}

variable "size" {
  default = "8Gi"
}

variable "chart_version"{
  default = "9.2.7"
}