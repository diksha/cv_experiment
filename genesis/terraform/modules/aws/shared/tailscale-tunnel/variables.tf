variable "deplomyment_identifier" {
  type    = string
  default = "tailscale"
}

variable "target_namespace" {
  type = string
  default = "tailscale"
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
  description = "tailscale auth key"
}