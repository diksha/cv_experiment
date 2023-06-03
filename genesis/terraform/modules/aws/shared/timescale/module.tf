locals {
  oidc_provider = replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")
  backup_bucket_name = "${var.db_identifier}-timescale-backup-${var.target_account_id}"
  service_account_name = "${var.db_identifier}-timescale"
}

module "backup_bucket" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = local.backup_bucket_name
  enable_versioning = false
}


resource "aws_iam_policy" "backup_bucket_access_policy" {
  name        = "s3-${local.backup_bucket_name}"
  path        = "/"
  description = "Timescale backup bucket IRSA"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:*",
        ]
        Effect   = "Allow"
        Resource = [
          "arn:aws:s3:::${local.backup_bucket_name}",
          "arn:aws:s3:::${local.backup_bucket_name}/*"
        ]
      },
    ]
  })
}

module "irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = var.db_identifier
  serviceaccount = local.service_account_name
  oidc_url       = local.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.target_account_id}:oidc-provider/${local.oidc_provider}"
  policy_arns    = [
       resource.aws_iam_policy.backup_bucket_access_policy.arn,
  ]
}


resource "helm_release" "primary" {
  name             = "${var.db_identifier}-timescale"
  namespace        = var.db_identifier
  chart            = "charts/timescaledb-single"
  repository       = "https://charts.timescale.com/"
  version          = var.chart_version
  wait             = var.wait_for_db
  timeout          = 300
  replace          = false
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    BACKUP_BUCKET_NAME = local.backup_bucket_name
    SERVICE_ACCOUNT_NAME = local.service_account_name
    IRSA_ROLE_ARN = module.irsa.arn
    REGION = var.primary_region
    IRSA_ROLE_NAME = "s3-${local.backup_bucket_name}"
    })
  ]
}


