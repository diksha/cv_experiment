locals {
  oidc_provider     = replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")
  target_account_id = var.context.target_account_id
  primary_region    = var.context.primary_region
  environment       = var.context.environment
}

module "portal" {
  name               = "portal"
  context            = var.context
  oidc_provider      = local.oidc_provider
  starting_image_tag = var.portal_starting_image_tag
  domain             = var.portal_domain
  voxel_portal_bucket_arn = module.voxel_portal_bucket.bucket_arn
  voxel_bucket_portal_name = module.voxel_portal_bucket.bucket_name

  source = "./modules/portal"
  providers = {
    aws      = aws
    aws.east = aws.east
  }
}

module "voxel_portal_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = local.target_account_id
  primary_region    = local.primary_region
  bucket_name       = "voxel-portal-${local.environment}"
  enable_versioning = true
  noncurrent_days   = 365
}

resource "aws_s3_bucket_policy" "voxel_portal_bucket_policy" {
  bucket = module.voxel_portal_bucket.bucket_id
  policy = jsonencode({
    Statement = [
      {
        Sid = "ReadWrite Permission"
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
          "s3:PutObject",
          "s3:RestoreObject",
          "s3:AbortMultipartUpload",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::voxel-portal-${local.environment}",
          "arn:aws:s3:::voxel-portal-${local.environment}/*"
        ],
        Principal = {
          AWS = [
            "arn:aws:iam::203670452561:role/BuildkiteAccess",
          ]
        }
      },
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
          "arn:aws:s3:::voxel-portal-${local.environment}",
          "arn:aws:s3:::voxel-portal-${local.environment}/*"
        ],
        Principal = {
          AWS = [
            "arn:aws:iam::203670452561:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6",
            # sematic-worker irsa for EKS Pod in Prime Account DevOps VPC Jenkins Cluster
            "arn:aws:iam::203670452561:role/irsa-tt6b0fvxadvn",
          ]
        }
      }
    ]
    Version = "2012-10-17"
  })
}

# Secrets Store CSI Driver and aws provider
resource "helm_release" "csi_secrets_store_provider_aws" {
  count            = 1
  chart            = "csi-secrets-store-provider-aws"
  repository       = "https://aws.github.io/eks-charts"
  name             = "csi-secrets-store-provider-aws"
  namespace        = "kube-system"
  version          = "0.0.3"
  atomic           = true
  create_namespace = false
  force_update     = true
}

resource "aws_s3_bucket_cors_configuration" "voxel_portal_bucket_cors" {
  bucket = module.voxel_portal_bucket.bucket_id
  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET"]
    allowed_origins = [local.environment == "production" ? "https://app.voxelai.com" : "https://app.staging.voxelplatform.com"]
    expose_headers  = ["Access-Control-Allow-Origin"]
    max_age_seconds = 3000
  }
}
