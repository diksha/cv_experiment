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

inputs = {
  eks_cluster_name                          = dependency.perception_infra.outputs.eks_cluster_name
  target_account_id                         = dependency.commons.outputs.staging_account_id
  services_root_ca                          = dependency.commons.outputs.services_root_ca
}