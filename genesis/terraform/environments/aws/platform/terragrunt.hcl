include "root" {
  path = find_in_parent_folders()
}

terraform {
  source = "${get_parent_terragrunt_dir()}/../../modules/aws//platform"
}

dependency "account_management" {
  config_path = "../account-management"
}

inputs = {
  target_account_id = dependency.account_management.outputs.platform_account_id
  specific_vars = {}
}