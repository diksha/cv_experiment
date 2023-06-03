resource "aws_organizations_account" "account" {
  name      = var.account_name
  email     = var.account_email
  role_name = var.sso_admin_role_name
  lifecycle {
    prevent_destroy = true
    ignore_changes  = all
  }
}

data "aws_ssoadmin_instances" "instance" {}

data "aws_ssoadmin_permission_set" "admin_access" {
  instance_arn = tolist(data.aws_ssoadmin_instances.instance.arns)[0]
  name         = var.sso_admin_role_name
}

data "aws_identitystore_group" "super_admins" {
  identity_store_id = tolist(data.aws_ssoadmin_instances.instance.identity_store_ids)[0]
  filter {
    attribute_path  = "DisplayName"
    attribute_value = var.sso_admin_group_name
  }
}

resource "aws_ssoadmin_account_assignment" "super_admin_assignment" {
  instance_arn       = data.aws_ssoadmin_permission_set.admin_access.instance_arn
  permission_set_arn = data.aws_ssoadmin_permission_set.admin_access.arn
  principal_id       = data.aws_identitystore_group.super_admins.group_id
  principal_type     = "GROUP"
  target_id          = aws_organizations_account.account.id
  target_type        = "AWS_ACCOUNT"
}

