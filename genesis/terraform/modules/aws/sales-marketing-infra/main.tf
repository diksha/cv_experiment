locals {
  sales_marketing_alias     = "sales-marketing"
  fivetran_source_ip_cidr   = "35.234.176.144/29"
  platform_primary_vpc_cidr = "10.1.0.0/16"
}

module "sales_marketing_vpc" {
  providers = {
    aws = aws
  }
  source             = "../shared/vpc-and-subnets"
  target_account_id  = var.target_account_id
  environment        = var.environment
  vpc_cidr_root      = var.sales_marketing_vpc_cidr_root
  vpc_name           = local.sales_marketing_alias
  enable_nat_gateway = false
}


module "sales_marketing_db" {
  source                                  = "../shared/rds-postgres"
  account_id                              = var.target_account_id
  db_identifier                           = "${local.sales_marketing_alias}-${var.environment}-postgres"
  subnet_ids                              = module.sales_marketing_vpc.public_subnet_ids
  db_instance_size                        = var.db_instance_size
  dev_mode                                = lower(var.environment) == "production" ? false : true
  is_publically_accessible                = true
  environment                             = var.environment
  vpc_id                                  = module.sales_marketing_vpc.vpc_id
  ingress_cidr_blocks                     = concat(module.sales_marketing_vpc.private_subnet_cidrs, [local.platform_primary_vpc_cidr, local.fivetran_source_ip_cidr])
  live_replica                            = false
  monitoring_interval                     = 0
  multi_az                                = true
  performance_insights_enabled            = false
  replica_vpc_cidr_root                   = var.sales_marketing_RDS_replica_cidr_root
  replica_region                          = var.sales_marketing_RDS_replica_region
  replica_vpc_name                        = "${local.sales_marketing_alias}-rds-replica-vpc"
  aws_notifications_slack_app_webhook_url = var.aws_notifications_slack_app_webhook_url
  create_rds_alarms_sns_and_slack         = true
  memory_freeable_too_low_threshold       = 64000000
  memory_swap_usage_too_high_threshold    = 512000000
}
