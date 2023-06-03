include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "${get_parent_terragrunt_dir()}/../../modules/aws//nucleus"
}

dependency "commons" {
  config_path = "${get_parent_terragrunt_dir()}/commons"
}

inputs = {
  account_id                              = dependency.commons.outputs.production_account_id
  aws_notifications_slack_app_webhook_url = dependency.commons.outputs.aws_notifications_slack_app_webhook_url
}