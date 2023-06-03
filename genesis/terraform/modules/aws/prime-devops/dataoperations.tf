resource "aws_iam_policy" "dataoperations_access" {
  name = "DataoperationsAccess"
  description = "Data Operations access permissions"
  policy = data.aws_iam_policy_document.dataoperations_access.json
}

data "aws_iam_policy_document" "dataoperations_access" {
  statement {
    sid = "DataoperationsS3Read"
    actions = [
      "s3:ListBucketMultipartUploads",
      "s3:ListBucketVersions",
      "s3:ListBucket",
      "s3:ListAllMyBuckets",
      "s3:ListMultipartUploadParts",
      "s3:GetObjectAcl",
      "s3:GetObject",
      "s3:GetBucketAcl",
      "s3:GetObjectVersionAcl",
      "s3:GetObjectVersion",
      "s3:GetBucketLocation"
    ]
    resources = [
      "arn:aws:s3:::${module.voxel_temp_dataops_bucket.bucket_name}",
      "arn:aws:s3:::${module.voxel_temp_dataops_bucket.bucket_name}/*",
    ]
  }
}
