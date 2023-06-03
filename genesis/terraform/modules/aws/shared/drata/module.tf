locals {
    drata_aws_account_id = "269135526815"
    drata_external_id = "ded65a9d-e173-4e21-b916-b1992d7e76d7"
}

resource "aws_iam_role" "drata_external_access" {
  name = "DrataAutopilotRole"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = "AssumeRole"
        Principal = {
          AWS = [
            local.drata_aws_account_id
          ]
        }
        Condition = {
            "StringEquals" = {
                "sts:ExternalId": local.drata_external_id
            }
        }
      }
    ]
  })
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/SecurityAudit", 
  ]
  max_session_duration = 3600
}

