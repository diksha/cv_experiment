moved {
  from = module.product_base_setup.module.vpc_primary
  to = module.vpc.module.vpc_primary
}

locals {
  perception_alias          = "perception"
  product_alias             = "product"
  platform_primary_vpc_cidr = "10.1.0.0/16"
  root_account_vpc_cidr     = "172.16.0.0/22"
}

module "vpc" {
  providers = {
    aws = aws
  }
  source             = "../shared/vpc-and-subnets"
  target_account_id  = var.target_account_id
  environment        = var.environment
  vpc_cidr_root      = var.product_vpc_cidr_root
  vpc_name           = local.product_alias
  enable_nat_gateway = true
  public_subnet_tags = {
    "kubernetes.io/role/elb"                       = "1"
    "kubernetes.io/cluster/${local.product_alias}" = "shared"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"              = "1"
    "kubernetes.io/cluster/${local.product_alias}" = "shared"
  }
}

module "product_base_setup" {
  providers = {
    aws = aws
  }
  source                                  = "../shared/base-env"
  account_id                              = var.target_account_id
  primary_region                          = var.primary_region
  environment                             = var.environment
  dev_mode                                = false
  eks_should_create_standard_node_group   = true
  eks_should_create_gpu_node_group        = false
  vpc_cidr_root                           = var.product_vpc_cidr_root
  vpc_name                                = local.product_alias
  vpc_id                                  = module.vpc.vpc_id
  private_subnets                         = module.vpc.private_subnet_ids
  security_group_ids                      = module.vpc.vpc_security_group_ids
  cluster_name                            = local.product_alias
  eks_cpu_instance_types                  = split(",", var.portal_eks_cpu_instance_types_comma_separated)
  eks_default_max_instance_count          = var.eks_default_max_instance_count
  eks_default_disk_size                   = 200
  cluster_version                         = "1.24"
  aws_notifications_slack_app_webhook_url = var.aws_notifications_slack_app_webhook_url
  create_alarm                            = lower(var.environment) == "production" ? true : false
}

resource "aws_vpc_endpoint" "s3_gateway_primary_region" {
  vpc_id       = module.vpc.vpc_id
  service_name = "com.amazonaws.${var.primary_region}.s3"
  route_table_ids = concat(
    tolist(module.vpc.private_route_table_ids),
    tolist(module.vpc.public_route_table_ids),
  )
}

module "product_observability" {
  providers = {
    aws = aws
  }
  source                   = "../shared/eks-observability"
  account_id               = var.target_account_id
  cluster_name             = module.product_base_setup.eks_cluster_name
  grafana_url              = var.grafana_url
  grafana_irsa_arn         = var.grafana_irsa_arn
  grafana_api_key          = var.grafana_api_key
  observability_identifier = "${var.observability_identifier}-${local.product_alias}"
  register_with_grafana    = true
  aws_region               = var.primary_region
  install_gpu_components   = false
}

module "portal_db" {
  source                                  = "../shared/rds-postgres"
  account_id                              = var.target_account_id
  db_identifier                           = "portal-${var.environment}-postgres"
  subnet_ids                              = module.vpc.private_subnet_ids
  db_instance_size                        = var.db_instance_size
  dev_mode                                = lower(var.environment) == "production" ? false : true
  is_publically_accessible                = false
  environment                             = var.environment
  vpc_id                                  = module.vpc.vpc_id
  ingress_cidr_blocks                     = concat(module.vpc.private_subnet_cidrs, [local.platform_primary_vpc_cidr, local.root_account_vpc_cidr])
  replica_vpc_cidr_root                   = var.portal_RDS_replica_cidr_root
  replica_region                          = var.portal_RDS_replica_region
  replica_vpc_name                        = "portal-rds-replica-vpc"
  performance_insights_retention_period   = var.performance_insights_retention_period
  create_rds_alarms_sns_and_slack         = lower(var.environment) == "production" ? true : false
  backup_replica                          = lower(var.environment) == "production" ? true : false
  aws_notifications_slack_app_webhook_url = var.aws_notifications_slack_app_webhook_url
}

module "portal_redis" {
  source              = "../shared/redis-cluster"
  cluster_name        = "portal-${var.environment}-redis"
  subnet_ids          = module.vpc.private_subnet_ids
  description         = "redis cluster for portal"
  node_type           = var.redis_instance_type
  cluster_size        = var.redis_cluster_size
  vpc_id              = module.vpc.vpc_id
  vpc_cidr_root       = var.product_vpc_cidr_root
  ingress_cidr_blocks = module.vpc.private_subnet_cidrs #local.portal_private_subnets
}

module "autocert_mtls" {
  source      = "../shared/eks-autocert-mtls"
  root_ca     = var.services_root_ca
  common_name = "Product Services Intermediate CA - ${local.product_alias}.${var.environment}"
}
