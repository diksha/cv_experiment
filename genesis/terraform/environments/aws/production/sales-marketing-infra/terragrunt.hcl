include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "${get_parent_terragrunt_dir()}/../../modules/aws//sales-marketing-infra"
}

dependency "commons" {
  config_path = "${get_parent_terragrunt_dir()}/commons"
}

inputs = {
  grafana_url                               = dependency.commons.outputs.grafana_url
  grafana_api_key                           = dependency.commons.outputs.grafana_api_key
  grafana_irsa_arn                          = dependency.commons.outputs.grafana_irsa_arn
  target_account_id                         = dependency.commons.outputs.production_account_id
  aws_notifications_slack_app_webhook_url   = dependency.commons.outputs.aws_notifications_slack_app_webhook_url
}