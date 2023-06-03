remote_state {
  backend = "gcs"
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite"
  }
  config = {
    bucket                        = local.bucket_name
    project                       = "voxel-devops-production"
    location                      = "US"
    prefix                         = "/${local.values.state_prefix}"
    encryption_key                = local.secret_vars.encryption_key
  }
}

inputs = merge({
  context = {
    primary_region = "us-west-2"
    environment = local.values.variables.environment
  }
},merge(merge(local.secret_vars.variables,local.standard_vars),local.values.variables))

locals {
  bucket_name = "terraform-sl1"
  standard_vars = {
    root_account_id = "667031391229"
    primary_region = "us-west-2"
    route53_zone_id = "Z08614703PK7668A4P1AC"
    admin_role_name = "AdministratorAccess"
    sso_group_name = "Administrators"
    root_domain = "voxelai.com"
  }
  secret_vars = yamldecode(sops_decrypt_file("${get_terragrunt_dir()}/secrets.sops.yaml"))
  values = yamldecode(file("${get_terragrunt_dir()}/values.yaml"))
}

