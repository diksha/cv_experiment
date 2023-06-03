resource "aws_s3_bucket_policy" "voxel_storage_bucket_policy" {
  bucket = "voxel-storage"
  policy = jsonencode({
    Statement = [
      {
        Sid = "Read Only Permission"
        Action = [
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:ListBucket",
          "s3:ListMultipartUploadParts",
          "s3:GetObjectAcl",
          "s3:GetObject",
          "s3:GetBucketAcl",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersion",
          "s3:GetBucketLocation",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-storage",
          "arn:aws:s3:::voxel-storage/*"
        ],
        Principal = {
          AWS = [
            "arn:aws:iam::667031391229:role/buildkite_access_assumable_role",
            # This role is added to allow to build and push containers when using production admin role for perception deployments.
            "arn:aws:iam::360054435465:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_AdministratorAccess_1beb57eaacf7543c",
            # Allows production runners read access to this bucket to download models
            "arn:aws:iam::360054435465:role/irsa-mt02f7b48b9g",
          ]
        }
      }
      ]
    Version = "2012-10-17"
    })
}

resource "aws_s3_bucket_policy" "voxel_raw_logs_bucket_policy" {
  bucket = "voxel-raw-logs"
  policy = jsonencode({
    Statement = [
      {
        Sid = "CopyObjectPermission"
        Action = [
          "s3:PutObject",
          "s3:PutObjectTagging"
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-raw-logs",
          "arn:aws:s3:::voxel-raw-logs/*"
        ],
        Principal = {
          AWS = [
            # Portal Production API Service IRSA.
            "arn:aws:iam::360054435465:role/irsa-91a0485uzk9b",
          ]
        }
      }
      ]
    Version = "2012-10-17"
    })
}
