variable "context" {
  type = object({
    eks_cluster_name = string
    environment = string
    target_account_id = string
    primary_region = string
  })
}

variable "name" {
  type    = string
}

variable "oidc_provider" {
  type = string
}

variable "extra_policy_arns" {
  type = list(string)
  default = []
}

variable "initial_image_tag" {
  type = string
}

variable "tls_sans" {
  type = list(string)
  default = []
}
