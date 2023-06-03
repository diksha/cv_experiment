output eks_cluster_name {
  value       = module.product_base_setup.eks_cluster_name
  sensitive   = false
}

output vpc {
  value = {
    id = module.vpc.vpc_id
    private_subnet_ids = module.vpc.private_subnet_ids
  }
}
