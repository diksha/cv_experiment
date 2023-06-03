variable "account_name" {
  type = string
}

variable "account_email" {
  type = string
}

variable "root_account_id" {
  type = string
}

variable "sso_admin_role_name" {
  type = string
  default = "AdministratorAccess"
}

variable "sso_admin_group_name" {
  type = string
  default = "Administrators"
}