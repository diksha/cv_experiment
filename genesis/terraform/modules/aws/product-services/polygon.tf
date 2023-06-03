locals {
  polygon_dns_names = {
    "production": "polygon.voxelplatform.com"
    "staging": "polygon.staging.voxelplatform.com"
  }
}

module "polygon" {
  count = local.environment != "production" ? 1 : 0
  source = "./modules/generic-service"
  name = "polygon"
  context = var.context
  initial_image_tag = "31c15f0f70a7641dda87dba945951224df4d30f664958813b6cc14dfcf165543"
  oidc_provider = local.oidc_provider
  tls_sans = [local.polygon_dns_names[local.environment]]
}

resource "kubernetes_service_v1" "polygon_external" {
  count = local.environment != "production" ? 1 : 0
  metadata {
    name = "polygon-external"
    namespace = module.polygon[0].namespace
    annotations = {
      "service.beta.kubernetes.io/aws-load-balancer-scheme" = "internet-facing"
      "service.beta.kubernetes.io/aws-load-balancer-type" = "external"
      "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type" = "ip"
      "service.beta.kubernetes.io/aws-load-balancer-additional-resource-tags" = "map-migrated=d-server-00swbp99drezfh"
    }
  }
  spec {
    selector = {
      "app.kubernetes.io/name" = "polygon"
    }
    port {
      port        = 443
      target_port = 8443
    }
    type = "LoadBalancer"
  }
}

# S3 Bucket
module "polygon_graph_config_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = local.target_account_id
  primary_region    = local.primary_region
  bucket_name       = "voxel-${local.environment}-polygon-graph-configs"
  enable_versioning = true
  noncurrent_days   = 1
  expiration_days   = 180
}

# S3 bucket policy
data "aws_iam_policy_document" "polygon_graph_config_bucket_access_policy" {
  statement {
    sid = "AllowS3"
    principals {
      type = "AWS"
      identifiers = ["arn:aws:iam::203670452561:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6"]
    }
    effect = "Allow"
    actions = [
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
    ]
    resources = [
      module.polygon_graph_config_bucket.bucket_arn,
      "${module.polygon_graph_config_bucket.bucket_arn}/*",
    ]
  }
}

resource "aws_s3_bucket_policy" "polygon_graph_config_bucket_access_policy"  {
  bucket = module.polygon_graph_config_bucket.bucket_id
  policy = data.aws_iam_policy_document.polygon_graph_config_bucket_access_policy.json
}
