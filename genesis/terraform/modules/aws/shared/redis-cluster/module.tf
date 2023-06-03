# based off of https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/elasticache_replication_group

resource "aws_elasticache_subnet_group" "ec_subnet_group" {
  name       = "${var.cluster_name}-cache-subnet-group"
  subnet_ids = var.subnet_ids
}

resource "aws_elasticache_replication_group" "redis" {
  automatic_failover_enabled  = var.cluster_size > 1 ? true : false
  #preferred_cache_cluster_azs = var.cluster_azs
  replication_group_id        = var.cluster_name
  description                 = var.description
  node_type                   = var.node_type
  num_cache_clusters          = var.cluster_size
  engine                      = "redis"
  engine_version              = "6.2"
  parameter_group_name        = "default.redis6.x"
  port                        = 6379
  multi_az_enabled            = var.cluster_size > 1 ? true : false
  subnet_group_name           = resource.aws_elasticache_subnet_group.ec_subnet_group.name
  security_group_ids          = [module.aws_elasticache_internal_access_sg.security_group_id]
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
}

module "aws_elasticache_internal_access_sg" {
  source              = "terraform-aws-modules/security-group/aws//modules/redis"
  name                = "${var.cluster_name}-sg"
  description         = "Security group to access redis cluster ${var.cluster_name} from ${join(",", var.subnet_ids)}"
  vpc_id              = var.vpc_id
  ingress_cidr_blocks = var.ingress_cidr_blocks
}
