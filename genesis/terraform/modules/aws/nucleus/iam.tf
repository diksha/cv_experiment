locals {
  prime_jenkins_eks_oidc_provider = "oidc.eks.us-west-2.amazonaws.com/id/7F736585AF1A879653C8E0819543D12E"
}

resource "aws_iam_policy" "cloudfront_policy_buildkite_role" {
  name        = "cloudfront_policy"
  path        = "/"
  description = "Cloudfront policy for Buildkite role"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "cloudfront:CreateInvalidation",
        ]
        Effect   = "Allow"
        Resource = "*"
      },
    ]
  })
}

resource "aws_iam_role" "buildkite_access_assumable_role" {
  name = "buildkite_access_assumable_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = "AssumeRole"
        Principal = {
          AWS = [
            "arn:aws:iam::203670452561:role/BuildkiteAccess",
          ]
        }
      },
    ]
  })
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess",
    aws_iam_policy.cloudfront_policy_buildkite_role.arn
  ]
  max_session_duration = 43200
}
