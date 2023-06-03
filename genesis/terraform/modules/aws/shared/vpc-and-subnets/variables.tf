variable "vpc_cidr_root" {
  description = "Give a single number 'X' denoting the 10.X.0.0/16 CIDR that should be used for the VPC"
  type        = string
}

variable "az_count" {
  description = "number of azs to create the vpc and subnets in. defaults to 2 and only accepts 2 or for any non-2 number will be 3"
  type        = number
  default     = 2
}

variable "primary_region" {
  type    = string
  default = "us-west-2"
}

variable "environment" {
  description = "Additional tags for the private subnets"
  type        = string
}

variable "private_subnet_tags" {
  description = "Additional tags for the private subnets"
  type        = map(string)
  default     = {}
}

variable "public_subnet_tags" {
  description = "Additional tags for the public subnets"
  type        = map(string)
  default     = {}
}

variable "tags" {
  description = "A map of tags to add to all resources"
  type        = map(string)
  default     = {}
}

variable "vpc_name" {
  type = string
}

variable "target_account_id" {
  type = string
}


variable "enable_nat_gateway" {
  default = true
}