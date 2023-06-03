data "aws_iam_policy_document" "dev_account_assummable_role_assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]
    sid = "DeveloperAccessAssumeRole"
    principals {
      type = "AWS"
      identifiers = ["arn:aws:iam::203670452561:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6"]
    }
  }
}

resource "aws_iam_role" "dev_account_assumable_role" {
  provider = aws.development
  name = "developer_access_assumable_role"
  assume_role_policy = data.aws_iam_policy_document.dev_account_assummable_role_assume_role_policy.json
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AdministratorAccess",
  ]
}