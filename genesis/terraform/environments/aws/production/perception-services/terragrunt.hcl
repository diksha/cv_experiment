include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "${get_parent_terragrunt_dir()}/../../modules/aws//perception-services"
}

dependency "commons" {
  config_path = "${get_parent_terragrunt_dir()}/commons"
}


dependency "perception_infra" {
  config_path = "../perception-infra"
}

dependency "platform" {
  config_path = "${get_parent_terragrunt_dir()}/platform"
}

inputs = {
  eks_cluster_name                          = dependency.perception_infra.outputs.eks_cluster_name
  states_events_bucket_arn                  = dependency.perception_infra.outputs.states_events_bucket_arn
  target_account_id                         = dependency.commons.outputs.production_account_id
  prime_account_id                          = dependency.commons.outputs.prime_account_id
  argo_server_domain                        = "argo.voxelplatform.com"
  argo_username                             = dependency.platform.outputs.argo_admin_username
  argo_password                             = dependency.platform.outputs.argo_admin_password
  services_root_ca                          = dependency.commons.outputs.services_root_ca
}