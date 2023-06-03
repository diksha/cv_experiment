locals {
    scale_ai_aws_account_id = "307185671274"
    scale_ai_external_id = "61831618cf0734003a62eb48"
}

resource "aws_iam_policy" "scale_ai_access" {
  name        = "ScaleAIAccess"
  description = "ScaleAI Access"
  policy = jsonencode({
    Statement = [
      {
        Sid = "ReadAccess"
        Action = [
            "s3:GetObject",
            "s3:ListBucket"
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-logs/*",
          "arn:aws:s3:::voxel-logs",
          "arn:aws:s3:::voxel-datasets/*",
          "arn:aws:s3:::voxel-datasets"
        ],
      },
    ]
    Version = "2012-10-17"
  })
}


resource "aws_iam_role" "scale_ai_external_access" {
  # This name is required by Scale AI.
  name = "ScaleAI-Integration"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = "AssumeRole"
        Principal = {
          AWS = [
            local.scale_ai_aws_account_id
          ]
        }
        Condition = {
            "StringEquals" = {
                "sts:ExternalId": local.scale_ai_external_id
            }
        }
      }
    ]
  })
  managed_policy_arns = [
    aws_iam_policy.scale_ai_access.arn, 
  ]
  max_session_duration = 3600
}

