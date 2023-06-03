output "eks_cluster_name" {
  value = module.perception_base_setup.eks_cluster_name
}

output "states_events_bucket_arn" {
  value = module.states_events_bucket.bucket_arn
}