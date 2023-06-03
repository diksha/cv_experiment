variable "deplomyment_identifier" {
  type    = string
  default = "cloudflared"
}

variable "target_namespace" {
  type = string
  default = "kube-system"
}

variable "cluster_name" {
  type = string
}

variable "primary_region" {
  type = string
}

variable "tunnel_token" {
  type = string
  sensitive = true
  description = "(optional) describe your variable"
}