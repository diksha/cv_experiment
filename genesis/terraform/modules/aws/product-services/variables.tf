variable "context" {
  type = object({
    environment = string
    target_account_id = number
    eks_cluster_name = string
    primary_region = string

    vpc = object({
      id = string
      private_subnet_ids = list(string)
    })
  })
}

variable "name" {
  type    = string
  default = "product"
}

variable "argo" {
  type = object({
    domain = string
    port = optional(number, 443)
    username = string
    password = string
  })
  sensitive = true
}

variable "portal_starting_image_tag" {
  type = string
}

variable "portal_secret_arn" {
  type = string
}

variable "portal_domain" {
  type = string
}

variable "slack_notification_sns_topic_arn" {
  type = string
}

variable "services_root_ca" {
  type = object({
    cert_pem = string
    key_pem = string
    password = string
  })
  sensitive = true
}
