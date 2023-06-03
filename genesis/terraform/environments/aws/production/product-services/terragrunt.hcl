include "root" {
  path = find_in_parent_folders()
  merge_strategy = "deep"
}

terraform {
  source = "${get_parent_terragrunt_dir()}/../../modules/aws//product-services"
}

dependency "commons" {
  config_path = "${get_parent_terragrunt_dir()}/commons"
}

dependency "platform" {
  config_path = "${get_parent_terragrunt_dir()}/platform"
}

dependency "nucleus" {
  config_path = "../nucleus"
}

dependency "product" {
  config_path = "../product-infra"
}

inputs = {
  context = {
    vpc                                       = dependency.product.outputs.vpc
    eks_cluster_name                          = dependency.product.outputs.eks_cluster_name
    target_account_id                         = dependency.commons.outputs.production_account_id
  }

  argo = {
    domain                                    = "argo.voxelplatform.com"
    username                                  = dependency.platform.outputs.argo_admin_username
    password                                  = dependency.platform.outputs.argo_admin_password
  }

  slack_notification_sns_topic_arn            = dependency.nucleus.outputs.slack_notification_sns_topic_arn
  portal_secret_arn                           = "arn:aws:secretsmanager:us-west-2:360054435465:secret:portal-PAo2Cu"

  services_root_ca                          = dependency.commons.outputs.services_root_ca
}
