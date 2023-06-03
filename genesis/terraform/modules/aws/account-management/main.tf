module "production_account_init" { # Was called account_init
  source          = "./account-init"
  account_name    = var.production["account_name"]
  account_email   = var.production["admin_email"]
  root_account_id = var.root_account_id
}

module "staging_account_init" { # Was called account_init
  source          = "./account-init"
  account_name    = var.staging["account_name"]
  account_email   = var.staging["admin_email"]
  root_account_id = var.root_account_id
}

module "development_account_init" { # Was called account_init
  source          = "./account-init"
  account_name    = var.development["account_name"]
  account_email   = var.development["admin_email"]
  root_account_id = var.root_account_id
}

module "prime_account_init" {
  source          = "./account-init"
  account_name    = var.prime["account_name"]
  account_email   = var.prime["admin_email"]
  root_account_id = var.root_account_id
}

module "galileo_account_init" {
  source          = "./account-init"
  account_name    = var.galileo["account_name"]
  account_email   = var.galileo["admin_email"]
  root_account_id = var.root_account_id
}

module "prime_engineering_access_group" {
  source               = "./sso-for-single-role"
  target_account_id    = module.prime_account_init.account_id
  target_iam_role_name = ""
  target_group_name    = "Engineering"
  primary_region       = var.primary_region
  create_inline_policy = false
  permission_set_name = "DeveloperAccess"
  create_customer_managed_policy = true
  customer_managed_policy_name = "DeveloperAccess"
}

module "production_edge_provisioning_access_group" {
  source               = "./sso-for-single-role"
  target_account_id    = module.production_account_init.account_id
  target_iam_role_name = ""
  target_group_name    = "EdgeProvisioning"
  primary_region       = var.primary_region
  create_inline_policy = false
  permission_set_name = "EdgeProvisioningAccess"
  create_group = true
  create_customer_managed_policy = true
  customer_managed_policy_name = "EdgeProvisioningAccess"
}

module "dataoperations_access_group" {
  source               = "./sso-for-single-role"
  target_account_id    = module.prime_account_init.account_id
  target_iam_role_name = ""
  target_group_name    = "DataOperations"
  primary_region       = var.primary_region
  create_inline_policy = false
  permission_set_name = "DataoperationsAccess"
  create_customer_managed_policy = true
  customer_managed_policy_name = "DataoperationsAccess"
}



module "platform_account_init" { # was called base-account-init
  source          = "./account-init"
  account_name    = var.platform["account_name"]
  account_email   = var.platform["admin_email"]
  root_account_id = var.root_account_id
}


resource "aws_iam_policy" "buildkite_access" {
  name        = "BuildkiteAccess"
  description = "Buildkite Access"
  policy = jsonencode({
    Statement = [
      {
        Sid = "ReadAccess"
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
          "arn:aws:s3:::voxel-storage/*",
          "arn:aws:s3:::voxel-storage",
        ],
      },
    ]
    Version = "2012-10-17"
  })
}


# Remove once portal moves out of root account.
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
            module.prime_account_init.account_id
          ]
        }
      }
    ]
  })
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/SecretsManagerReadWrite",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
    "arn:aws:iam::aws:policy/CloudFrontFullAccess",
    "arn:aws:iam::aws:policy/AmazonECS_FullAccess",
    "arn:aws:iam::aws:policy/AmazonElasticContainerRegistryPublicFullAccess",
    aws_iam_policy.buildkite_access.arn
  ]
  max_session_duration = 43200
}