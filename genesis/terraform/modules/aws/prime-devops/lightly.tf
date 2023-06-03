locals {
    lightly_aws_account_id = "916419735646"
    lightly_external_id = "bOMq4pBn2ESlsmZNwvua"
}

resource "aws_iam_policy" "lightly_access" {
  name        = "LightlyAccess"
  description = "Lightly Access"
  policy = jsonencode({
    Statement = [
      {
        Sid = "ReadWriteAccessforOnPremWorker"
        Action = [
            "s3:GetObject",
            "s3:DeleteObject",
            "s3:PutObject",
            "s3:ListBucket"
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-lightly-input/*",
          "arn:aws:s3:::voxel-lightly-input",
          "arn:aws:s3:::voxel-lightly-output/*",
          "arn:aws:s3:::voxel-lightly-output",
        ],
        # DevOps VPC in Prime account for lightly workers.
        Condition = {
            "StringEquals" = {
                "aws:SourceVpc": ["vpc-05a6d8fdd7f00bb29"]
            }
        }
      },
      # Lightly needs list access as it list files on its server and then sends the list to worker.
      {
        Sid = "ListBucket"
        Action = [
          "s3:ListBucket",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-lightly-input/*",
          "arn:aws:s3:::voxel-lightly-input",
          "arn:aws:s3:::voxel-lightly-output/*",
          "arn:aws:s3:::voxel-lightly-output",
        ]
      },
      # Lightly relevant filenames read permission to the file which contains the other names, right now its named all.txt.
      {
        Sid = "GetRelevantFilenamesObject"
        Action = [
          "s3:GetObject",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-lightly-output/*/all.txt",
          "arn:aws:s3:::voxel-lightly-output/*/relevant_filenames.txt",
        ]
      },
      # Primary VPC in Platform Account for lightly UI S3 access via Warp client.
      {
        Sid = "GetObject"
        Action = [
          "s3:GetObject",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-lightly-output/*",
          "arn:aws:s3:::voxel-lightly-output",
        ],
        Condition = {
            "StringEquals" = {
                "aws:SourceVpc": ["vpc-0c989e41fab24e123"]
            }
        }
      },
    ]
    Version = "2012-10-17"
  })
}


resource "aws_iam_role" "lightly_external_access" {
  name = "LightlyExternalAccess"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = "AssumeRole"
        Principal = {
          AWS = [
            local.lightly_aws_account_id
          ]
        }
        Condition = {
            "StringEquals" = {
                "sts:ExternalId": local.lightly_external_id
            }
        }
      }
    ]
  })
  managed_policy_arns = [
    aws_iam_policy.lightly_access.arn, 
  ]
  max_session_duration = 43200
}
