resource "aws_iam_role" "edge_provisioning_assumable_role" {
  provider = aws.production
  name     = "edge_provisioning_assumable_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = "AssumeRole"
        Principal = {
          AWS = [
            "arn:aws:iam::${var.production_account_id}:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_EdgeProvisioningAccess_8368b5a60fc9cb08",
          ]
        }
      },
    ]
  })
  managed_policy_arns = [
    aws_iam_policy.edge_provisioining_access.arn
  ]
  max_session_duration = 43200
}

resource "aws_iam_policy" "edge_provisioining_access" {
  provider    = aws.production
  name        = "EdgeProvisioningAccess"
  path        = "/"
  description = "Edge Provisioning Policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid = "CreateTokenExchangeRole"
        Action = [
          "iam:AttachRolePolicy",
          "iam:CreatePolicy",
          "iam:CreateRole",
          "iam:GetPolicy",
          "iam:GetRole",
          "iam:PassRole"
        ]
        Effect = "Allow"
        Resource = [
          "arn:aws:iam::${var.production_account_id}:role/GreengrassV2TokenExchangeRole",
          "arn:aws:iam::${var.production_account_id}:policy/GreengrassV2TokenExchangeRoleAccess"
        ]
      },
      {
        Sid    = "CreateIoTResources",
        Effect = "Allow",
        Action = [
          "iot:AddThingToThingGroup",
          "iot:AttachPolicy",
          "iot:AttachThingPrincipal",
          "iot:CreateKeysAndCertificate",
          "iot:CreatePolicy",
          "iot:CreateRoleAlias",
          "iot:CreateThing",
          "iot:CreateThingGroup",
          "iot:DescribeEndpoint",
          "iot:DescribeRoleAlias",
          "iot:DescribeThingGroup",
          "iot:GetPolicy"
        ],
        Resource = "*"
      },
      {
        Sid    = "DeployDevTools",
        Effect = "Allow",
        Action = [
          "greengrass:CreateDeployment",
          "iot:CancelJob",
          "iot:CreateJob",
          "iot:DeleteThingShadow",
          "iot:DescribeJob",
          "iot:DescribeThing",
          "iot:DescribeThingGroup",
          "iot:GetThingShadow",
          "iot:UpdateJob",
          "iot:UpdateThing",
          "iot:UpdateThingShadow"
        ],
        Resource = "*"
      }
    ]
  })
}
