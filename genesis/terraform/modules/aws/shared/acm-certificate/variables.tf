variable "domain" {
  type = string
}

variable "route53_zone_id" {
  type = string
}

variable "include_domain_in_san" {
  type = bool
  default = false
  
}