include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "${get_parent_terragrunt_dir()}/../../modules/aws//commons"
}

dependency "platform" {
  config_path = "../platform"
}

dependency "account_management" {
  config_path = "../account-management"
}

inputs = {
  grafana_url                               = dependency.platform.outputs.grafana_url
  grafana_api_key                           = dependency.platform.outputs.grafana_api_key
  grafana_irsa_arn                          = dependency.platform.outputs.grafana_irsa_arn
  platform_account_id                       = dependency.account_management.outputs.platform_account_id
  production_account_id                     = dependency.account_management.outputs.production_account_id
  development_account_id                    = dependency.account_management.outputs.development_account_id
  prime_account_id                          = dependency.account_management.outputs.prime_account_id
  staging_account_id                        = dependency.account_management.outputs.staging_account_id
  galileo_account_id                        = dependency.account_management.outputs.galileo_account_id
}