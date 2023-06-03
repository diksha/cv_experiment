variable "cluster_name" {
  type = string
}

variable "cluster_azs" {
  type = list
  default = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "description" {
  type = string
}

variable "node_type" {
  type = string
}

variable "cluster_size" {
  type = number
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list
}

variable "vpc_cidr_root" {
  type = string
}

variable "ingress_cidr_blocks" {
  type = list
}