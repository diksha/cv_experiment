output "vpc_id" {
  value = module.vpc_primary.vpc_id
}

output "private_subnet_ids" {
  value =module.vpc_primary.private_subnets
}

output "public_subnet_ids" {
  value =module.vpc_primary.public_subnets
}

output "vpc_security_group_ids" {
  value = [module.vpc_primary.default_security_group_id]
}

output "private_subnet_cidrs" {
  value = module.vpc_primary.private_subnets_cidr_blocks
}

output "public_subnet_cidrs" {
  value = module.vpc_primary.public_subnets_cidr_blocks
}

output "private_route_table_ids" {
  value = module.vpc_primary.private_route_table_ids
}

output "public_route_table_ids" {
  value = module.vpc_primary.public_route_table_ids
}