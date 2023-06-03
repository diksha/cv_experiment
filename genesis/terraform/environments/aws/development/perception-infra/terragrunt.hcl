include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "${get_parent_terragrunt_dir()}/../../modules/aws//perception-infra"
}

dependency "commons" {
  config_path = "../commons"
}

inputs = {
  grafana_url                               = dependency.commons.outputs.grafana_url
  grafana_api_key                           = dependency.commons.outputs.grafana_api_key
  grafana_irsa_arn                          = dependency.commons.outputs.grafana_irsa_arn
  target_account_id                         = dependency.commons.outputs.development_account_id
}