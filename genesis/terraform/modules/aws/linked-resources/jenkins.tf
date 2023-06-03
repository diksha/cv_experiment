# resource "aws_s3_bucket_policy" "jenkins_artifacts_access" {
#   provider = aws.platform

#   bucket = var.jenkins_s3_artifacts_bucket_name
#   policy = data.aws_iam_policy_document.allow_access_from_other_accounts.json
# }

# data "aws_iam_policy_document" "allow_access_from_other_accounts" {
#   provider = aws.platform

#   statement {
#     principals {
#       type        = "AWS"
#       identifiers = [var.prime_account_id]
#     }
#     actions = [
#       "s3:GetObject",
#       "s3:PutObject",
#       "s3:PutObjectAcl"
#     ]
#     resources = [
#       "arn:aws:s3:::${var.jenkins_s3_artifacts_bucket_name}",
#       "arn:aws:s3:::${var.jenkins_s3_artifacts_bucket_name}/*"
#     ]
#     effect = "Allow"
#   }
# }
