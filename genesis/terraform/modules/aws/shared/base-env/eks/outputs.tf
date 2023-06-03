output "storage_class_name" {
  value = "gp3-retain"
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_primary_security_group_id" {
  value = module.eks.cluster_primary_security_group_id
}

output "cluster_security_group_id" {
  value = module.eks.cluster_security_group_id
}


output "cluster_node_security_group_id" {
  value = module.eks.node_security_group_id
}
