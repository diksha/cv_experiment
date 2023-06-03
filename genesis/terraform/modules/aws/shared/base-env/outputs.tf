output "eks_cluster_name" {
  value = module.eks_services_cluster.cluster_name
}

output "eks_cluster_primary_security_group_id" {
  value = module.eks_services_cluster.cluster_primary_security_group_id
}
output "eks_cluster_security_group_id" {
  value = module.eks_services_cluster.cluster_security_group_id
}

output "eks_cluster_node_security_group_id" {
  value = module.eks_services_cluster.cluster_node_security_group_id
}
