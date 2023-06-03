include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "${get_parent_terragrunt_dir()}/../../modules/aws//linked-resources/"
}

dependency "platform" {
  config_path = "../platform"
}

dependency "commons" {
  config_path = "../commons"
}

inputs = {
  argo_domain                               = "argo.voxelplatform.com"
  argo_admin_username                       = dependency.platform.outputs.argo_admin_username
  argo_admin_password                       = dependency.platform.outputs.argo_admin_password
  grafana_url                               = dependency.platform.outputs.grafana_url
  grafana_api_key                           = dependency.platform.outputs.grafana_api_key
  grafana_irsa_arn                          = dependency.platform.outputs.grafana_irsa_arn
  platform_primary_vpc_id                   = dependency.platform.outputs.primary_vpc_id
  root_account_id                           = dependency.commons.outputs.root_account_id
  platform_account_id                       = dependency.commons.outputs.platform_account_id
  production_account_id                     = dependency.commons.outputs.production_account_id
  development_account_id                    = dependency.commons.outputs.development_account_id
  prime_account_id                          = dependency.commons.outputs.prime_account_id
  staging_account_id                        = dependency.commons.outputs.staging_account_id
  galileo_account_id                        = dependency.commons.outputs.galileo_account_id
}
