variable "server_domain"{

}

variable "server_port"{
  default = 443
}
variable "admin_username"{}

variable "admin_password"{
  sensitive = true
}

variable "cluster_name" {
  type      = string
}

variable "cluster_alias" {
  type      = string
  default = ""
}
