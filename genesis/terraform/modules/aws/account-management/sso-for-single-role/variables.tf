variable "target_iam_role_name" {}
variable "target_group_name" {}
variable "target_account_id" {}
variable "primary_region" {
  type = string
}

variable "create_inline_policy" {
  default = false
}

variable "permission_set_name" {
  type = string
}

variable "create_customer_managed_policy" {
  default = false
}

variable "customer_managed_policy_name" {
  default = ""
}

variable "create_group" {
  default = false
}