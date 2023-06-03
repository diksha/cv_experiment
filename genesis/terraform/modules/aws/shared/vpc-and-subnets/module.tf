module "vpc_primary" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "3.12.0"

  name = var.vpc_name
  cidr = "10.${var.vpc_cidr_root}.0.0/16"

  azs = var.az_count == 2 ? ["${var.primary_region}a", "${var.primary_region}b"] : [
    "${var.primary_region}a", "${var.primary_region}b", "${var.primary_region}c"
  ]
  private_subnets = [
    "10.${var.vpc_cidr_root}.0.0/20", "10.${var.vpc_cidr_root}.16.0/20", "10.${var.vpc_cidr_root}.32.0/20",
    "10.${var.vpc_cidr_root}.128.0/20", "10.${var.vpc_cidr_root}.144.0/20", "10.${var.vpc_cidr_root}.160.0/20"
  ]
  public_subnets = [
    "10.${var.vpc_cidr_root}.192.0/20",
    "10.${var.vpc_cidr_root}.208.0/20",
    "10.${var.vpc_cidr_root}.224.0/20"
  ]
  enable_dns_hostnames = true
  enable_nat_gateway   = var.enable_nat_gateway
  single_nat_gateway   = true
  enable_vpn_gateway   = false
  public_subnet_tags   = var.public_subnet_tags
  private_subnet_tags  = var.private_subnet_tags
  tags                 = merge(var.tags, { "Environment" = var.environment })
}
