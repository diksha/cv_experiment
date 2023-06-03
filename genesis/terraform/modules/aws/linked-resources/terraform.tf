locals {
  role_name = "TerraformAccess"
  max_session_duration = 3600
  atlantis_irsa_role_name = "irsa-yuv7c5xs7x6a"
  platform_account_admin_sso_role_name = "AWSReservedSSO_AdministratorAccess_6f030177475b118d"
  root_account_admin_sso_role_name = "AWSReservedSSO_AdministratorAccess_0431570e259d65f0"
  aws_managed_administrator_policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

data "aws_iam_role" "atlantis_irsa" {
  provider = aws.platform
  name = local.atlantis_irsa_role_name
}

data "aws_iam_role" "platform_account_admin_sso" {
  provider = aws.platform
  name = local.platform_account_admin_sso_role_name
}

data "aws_iam_role" "root_account_admin_sso" {
  provider = aws.root
  name = local.root_account_admin_sso_role_name
}


data "aws_iam_policy_document" "assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = [
        # Atlantis IRSA to be able to access required resources.
        data.aws_iam_role.atlantis_irsa.arn,
        # Allow platform administrators to be able to assume for local debugging.
        data.aws_iam_role.platform_account_admin_sso.arn,
        # Remove this once everyone is using atlantis.
        data.aws_iam_role.root_account_admin_sso.arn,
      ]
    }
  }
}

resource "aws_iam_role" "terraform_access_root" {
  provider = aws.root
  name = local.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy.json
  managed_policy_arns = [
    local.aws_managed_administrator_policy_arn,
  ]
  max_session_duration = local.max_session_duration
}

resource "aws_iam_role" "terraform_access_platform" {
  provider = aws.platform
  name = local.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy.json
  managed_policy_arns = [
    local.aws_managed_administrator_policy_arn,
  ]
  max_session_duration = local.max_session_duration
}

resource "aws_iam_role" "terraform_access_staging" {
  provider = aws.staging
  name = local.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy.json
  managed_policy_arns = [
    local.aws_managed_administrator_policy_arn,
  ]
  max_session_duration = local.max_session_duration
}

resource "aws_iam_role" "terraform_access_prime" {
  provider = aws.prime
  name = local.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy.json
  managed_policy_arns = [
    local.aws_managed_administrator_policy_arn,
  ]
  max_session_duration = local.max_session_duration
}

resource "aws_iam_role" "terraform_access_production" {
  provider = aws.production
  name = local.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy.json
  managed_policy_arns = [
    local.aws_managed_administrator_policy_arn,
  ]
  max_session_duration = local.max_session_duration
}

resource "aws_iam_role" "terraform_access_development" {
  provider = aws.development
  name = local.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy.json
  managed_policy_arns = [
    local.aws_managed_administrator_policy_arn,
  ]
  max_session_duration = local.max_session_duration
}

resource "aws_iam_role" "terraform_access_galileo" {
  provider = aws.galileo
  name = local.role_name
  assume_role_policy = data.aws_iam_policy_document.assume_role_policy.json
  managed_policy_arns = [
    local.aws_managed_administrator_policy_arn,
  ]
  max_session_duration = local.max_session_duration
}