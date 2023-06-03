data "aws_ssoadmin_instances" "instance" {}

resource "aws_ssoadmin_permission_set" "permission_set" {
  name             = var.permission_set_name
  description      = "Permission set for ${var.permission_set_name}"
  instance_arn     = tolist(data.aws_ssoadmin_instances.instance.arns)[0]
  relay_state      = "https://s3.console.aws.amazon.com/s3/home?region=${var.primary_region}#"
  session_duration = "PT12H"
}

data "aws_iam_policy_document" "policy" {
  statement {
    sid = "1"

    actions = [
      "sts:AssumeRole",
    ]

    resources = [
      "arn:aws:iam::${var.target_account_id}:role/${var.target_iam_role_name}"
    ]
  }
}

resource "aws_ssoadmin_permission_set_inline_policy" "example" {
  count = var.create_inline_policy ? 1 : 0
  inline_policy      = data.aws_iam_policy_document.policy.json
  instance_arn       = aws_ssoadmin_permission_set.permission_set.instance_arn
  permission_set_arn = aws_ssoadmin_permission_set.permission_set.arn
}

resource "aws_ssoadmin_customer_managed_policy_attachment" "example" {
  count = var.create_customer_managed_policy ? 1 : 0
  instance_arn       = aws_ssoadmin_permission_set.permission_set.instance_arn
  permission_set_arn = aws_ssoadmin_permission_set.permission_set.arn
  customer_managed_policy_reference {
    name = var.customer_managed_policy_name
  }
}

resource "aws_identitystore_group" "group" {
  count = var.create_group ? 1 : 0
  display_name      = var.target_group_name
  description       = var.target_group_name
  identity_store_id = tolist(data.aws_ssoadmin_instances.instance.identity_store_ids)[0]
}

data "aws_identitystore_group" "group" {
  identity_store_id = tolist(data.aws_ssoadmin_instances.instance.identity_store_ids)[0]
  filter {
    attribute_path  = "DisplayName"
    attribute_value = var.target_group_name
  }
  depends_on = [
    aws_identitystore_group.group
  ]
}

resource "aws_ssoadmin_account_assignment" "assignment" {
  instance_arn       = aws_ssoadmin_permission_set.permission_set.instance_arn
  permission_set_arn = aws_ssoadmin_permission_set.permission_set.arn
  principal_id       = data.aws_identitystore_group.group.group_id
  principal_type     = "GROUP"
  target_id          = var.target_account_id
  target_type        = "AWS_ACCOUNT"
}
