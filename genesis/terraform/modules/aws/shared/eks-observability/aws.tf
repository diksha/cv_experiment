locals {
  aws_service_account_name = "aws"
}

module "observability_irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = "observability"
  serviceaccount = local.aws_service_account_name
  oidc_url       = local.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.account_id}:oidc-provider/${local.oidc_provider}"
  policy_arns    = [aws_iam_policy.observability_access.arn, "arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess"]
}

# resource "helm_release" "cloudwatch_exporter" {
#   name             = "cloudwatch-exporter"
#   namespace        = helm_release.observability_stack.namespace
#   create_namespace = true
#   repository       = "https://prometheus-community.github.io/helm-charts"
#   chart            = "prometheus-cloudwatch-exporter"
#   version          = "0.19.2"
#   values = [
#     templatefile("${path.module}/ymls/cloudwatch-exporter.yml", {
#       PRIMARY_AWS_REGION   = var.aws_region
#       SERVICE_ACCOUNT_NAME = kubernetes_service_account.observability_aws.metadata[0].name
#     })
#   ]
# }

module "loki_bucket" {
  source            = "../s3-bucket"
  target_account_id = var.account_id
  primary_region    = var.aws_region
  bucket_name       = "loki-${var.account_id}-${var.cluster_name}"
  enable_versioning = true
  noncurrent_days = 1
  expiration_days   = 90
}

module "tempo_bucket" {
  source            = "../s3-bucket"
  target_account_id = var.account_id
  primary_region    = var.aws_region
  bucket_name       = "tempo-${var.account_id}-${var.cluster_name}"
  enable_versioning = true
  noncurrent_days = 1
  expiration_days   = 90
}

resource "aws_iam_policy" "observability_access" {
  name        = "logging-access-${var.cluster_name}"
  path        = "/"
  description = "Loki access for ${var.cluster_name}"
  policy      = data.aws_iam_policy_document.observability_access.json
}

data "aws_iam_policy_document" "observability_access" {
  statement {
    actions = [
      "s3:*",
    ]
    resources = [
      module.loki_bucket.bucket_arn,
      module.tempo_bucket.bucket_arn,
      "${module.loki_bucket.bucket_arn}/*",
      "${module.tempo_bucket.bucket_arn}/*"
    ]
    effect = "Allow"
  }

  statement {
    actions = [
      "dynamodb:*"
    ]
    resources = [
      "*"
    ]
    effect = "Allow"
  }
}


