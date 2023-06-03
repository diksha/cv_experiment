include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "${get_parent_terragrunt_dir()}/../../modules/aws//prime-devops"
}

dependency "commons" {
  config_path = "${get_parent_terragrunt_dir()}/commons"
}

inputs = {
  context = {
    accounts = dependency.commons.outputs.accounts
  }
  grafana_url                               = dependency.commons.outputs.grafana_url
  grafana_api_key                           = dependency.commons.outputs.grafana_api_key
  grafana_irsa_arn                          = dependency.commons.outputs.grafana_irsa_arn
  target_account_id                         = dependency.commons.outputs.prime_account_id
  services_root_ca                          = dependency.commons.outputs.services_root_ca
}
