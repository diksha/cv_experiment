include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "${get_parent_terragrunt_dir()}/../../modules/aws//galileo"
}

dependency "commons" {
  config_path = "${get_parent_terragrunt_dir()}/commons"
}

inputs = {
  target_account_id                         = dependency.commons.outputs.galileo_account_id
}