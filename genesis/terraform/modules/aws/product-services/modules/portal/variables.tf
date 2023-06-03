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
  default = "product"
}
variable "oidc_provider" {
  type = string
}

variable "starting_image_tag" {
  type = string
}

variable "domain" {
  type = string
}

variable "voxel_portal_bucket_arn" {
  type = string
}

variable "voxel_bucket_portal_name" {
  type = string
}
