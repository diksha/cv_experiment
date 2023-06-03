locals {
    infinity_ai_aws_account_canonical_id = "8d1ea9d6e3b61058b4c4551bfb056f58c40a02684301450ed80cd3062d467014"
    # we use the decoded form of this id because otherwise this module will perpetually show changes
    infinity_ai_aws_account_id = "arn:aws:iam::058982626209:root"
}

resource "aws_s3_bucket_policy" "allow_access_from_infinity_ai_account" {
  bucket = module.voxel_infinity_ai_shared_bucket.bucket_id
  policy = data.aws_iam_policy_document.allow_access_from_inifity_ai_account.json
}

data "aws_iam_policy_document" "allow_access_from_inifity_ai_account" {
  statement {
    principals {
      type        = "AWS"
      identifiers = [local.infinity_ai_aws_account_id]
    }
    actions = [
        "s3:ListBucket",
        "s3:ListMultipartUploadParts",
        "s3:ListBucketMultipartUploads",
        "s3:GetObject",
        "s3:GetObjectAcl",
        "s3:DeleteObject",
        "s3:PutObject",
        "s3:PutObjectAcl",
        "s3:AbortMultipartUpload"
    ]
    resources = [
      module.voxel_infinity_ai_shared_bucket.bucket_arn,
      "${module.voxel_infinity_ai_shared_bucket.bucket_arn}/*",
    ]
  }
}

