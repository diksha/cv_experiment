locals {
  engine               = "postgres"
  engine_version       = "14.3"
  family               = "postgres14"
  major_engine_version = "14"
  port                 = "5432"
  parameters = [
    {
      name  = "autovacuum"
      value = 1
    },
    {
      name  = "client_encoding"
      value = "utf8"
    },
    {
      apply_method = "pending-reboot"
      name         = "pg_stat_statements.track"
      value        = "ALL"
    },
    {
      apply_method = "pending-reboot"
      name         = "shared_preload_libraries"
      value        = "pg_stat_statements"
    },
    {
      apply_method = "pending-reboot"
      name         = "track_activity_query_size"
      value        = "2048"
    }
  ]
  tags = {
    Service     = var.db_identifier
    Environment = var.environment
  }
  db_option_group_tags = {
    "Sensitive" = "low"
  }
  db_parameter_group_tags = {
    "Sensitive" = "low"
  }
}

module "primary" {
  source = "terraform-aws-modules/rds/aws"
  version = "5.2.0"
  identifier                          = var.db_identifier
  engine                              = local.engine
  engine_version                      = local.engine_version
  family                              = local.family
  major_engine_version                = local.major_engine_version
  instance_class                      = var.db_instance_size
  allocated_storage                   = var.allocated_storage
  max_allocated_storage               = 1000
  publicly_accessible                 = var.is_publically_accessible
  apply_immediately                   = true
  db_name                             = replace(var.db_identifier, "-", "_")
  username                            = "postgres"
  port                                = local.port
  iam_database_authentication_enabled = true
  delete_automated_backups            = var.dev_mode
  storage_encrypted                   = !var.dev_mode

  // Networking
  create_db_subnet_group = var.existing_db_subnet == null
  db_subnet_group_name   = var.existing_db_subnet != null ? var.existing_db_subnet : "${var.db_identifier}-db"
  vpc_security_group_ids = [module.vpc_primary_public_access_sg.security_group_id]
  subnet_ids             = var.subnet_ids


  // Observability and HA
  multi_az                = !var.dev_mode && var.multi_az
  maintenance_window      = "Sun:00:00-Sun:03:00"
  backup_window           = "03:00-06:00"
  backup_retention_period = var.dev_mode ? 0 : 7
  skip_final_snapshot     = var.dev_mode
  deletion_protection     = !var.dev_mode
  copy_tags_to_snapshot   = !var.dev_mode

  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  create_cloudwatch_log_group     = true

  performance_insights_enabled          = var.performance_insights_enabled
  performance_insights_retention_period = var.performance_insights_retention_period
  create_monitoring_role                = true
  monitoring_interval                   = var.monitoring_interval
  monitoring_role_name                  = "${var.db_identifier}-rds-monitoring-role"

  parameters              = local.parameters
  db_option_group_tags    = local.db_option_group_tags
  db_parameter_group_tags = local.db_parameter_group_tags
  tags                    = local.tags
}

module "vpc_primary_public_access_sg" {
  source              = "terraform-aws-modules/security-group/aws//modules/postgresql"
  name                = "${var.db_identifier}-allow-access-for-cidrs"
  description         = "Security group to access RDS ${var.db_identifier} from ${join(",", var.ingress_cidr_blocks)}"
  vpc_id              = var.vpc_id
  ingress_cidr_blocks = var.ingress_cidr_blocks
}

module "replica_vpc" {
  count = !var.dev_mode && var.live_replica ? 1 : 0
  providers = {
    aws = aws.replica
  }
  source              = "../vpc-and-subnets"
  vpc_cidr_root       = var.replica_vpc_cidr_root
  primary_region      = var.replica_region
  environment         = var.environment
  vpc_name            = var.replica_vpc_name
  private_subnet_tags = { "Name" = "${var.db_identifier}-replica-private-subnet" }
  public_subnet_tags  = { "Name" = "${var.db_identifier}-replica-public-subnet" }
  target_account_id   = var.account_id
}

module "vpc_replica_public_access_sg" {
  count  = !var.dev_mode && var.live_replica ? 1 : 0
  source = "terraform-aws-modules/security-group/aws//modules/postgresql"
  providers = {
    aws = aws.replica
  }
  name                = "${var.db_identifier}-replica-allow-access-for-cidrs"
  description         = "Security group to access RDS ${var.db_identifier} replica from ${join(",", var.ingress_cidr_blocks)}"
  vpc_id              = module.replica_vpc[0].vpc_id
  ingress_cidr_blocks = var.ingress_cidr_blocks
}

module "kms" {
  count  = var.dev_mode ? 0 : 1
  source = "terraform-aws-modules/kms/aws"
  providers = {
    aws = aws.replica
  }
  description = "KMS key for cross region replica DB"
  aliases     = ["voxel/${var.db_identifier}-replica-postgres-db-kms"]
  tags        = { "Name" = "${var.db_identifier}-replica-db-backup-kms" }
}

module "replica" {
  count  = !var.dev_mode && var.live_replica ? 1 : 0
  source = "terraform-aws-modules/rds/aws"
  version = "5.2.0"
  providers = {
    aws = aws.replica
  }
  replicate_source_db = module.primary.db_instance_arn

  create_random_password = false
  identifier             = "${var.db_identifier}-replica"
  engine                 = local.engine
  engine_version         = local.engine_version
  family                 = local.family
  major_engine_version   = local.major_engine_version
  instance_class         = var.db_instance_size
  storage_encrypted      = true
  allocated_storage      = 200
  max_allocated_storage  = 1000
  kms_key_id             = module.kms[0].key_arn
  port                   = local.port

  multi_az = false

  create_db_subnet_group = true
  db_subnet_group_name   = "${var.db_identifier}-replica-db"
  vpc_security_group_ids = [module.vpc_replica_public_access_sg[0].security_group_id]
  subnet_ids             = module.replica_vpc[0].private_subnet_ids

  maintenance_window              = "Sun:00:00-Sun:03:00"
  backup_window                   = "03:00-06:00"
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  backup_retention_period = 0
  skip_final_snapshot     = true
  deletion_protection     = false

  parameters              = local.parameters
  db_option_group_tags    = local.db_option_group_tags
  db_parameter_group_tags = local.db_parameter_group_tags
  tags                    = local.tags
}

resource "aws_db_instance_automated_backups_replication" "backup_replica" {
  count                  = var.backup_replica ? 1 : 0
  source_db_instance_arn = module.primary.db_instance_arn
  kms_key_id             = module.kms[0].key_arn
  provider               = aws.replica
}
