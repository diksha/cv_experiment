locals {
  prime_jenkins_eks_oidc_provider = "oidc.eks.us-west-2.amazonaws.com/id/7F736585AF1A879653C8E0819543D12E"
}

// TODO: Rename this resource
resource "kubernetes_namespace" "runners_production" {
  metadata {
    name = "runners-${var.environment}"
  }
}

// TODO: Rename this resource
resource "kubernetes_secret" "runner_secrets_production" {
  metadata {
    name      = "runner-secrets"
    namespace = kubernetes_namespace.runners_production.metadata[0].name
  }
  data = {
    GOOGLE_APPLICATION_CREDENTIALS_BASE64_ENCODED = var.google_application_credentials_base64_encoded
    SENTRY_DSN                                    = var.sentry_dsn
  }
}

resource "kubernetes_service_account" "runners_sa" {
  metadata {
    name      = "runners-sa"
    namespace = kubernetes_namespace.runners_production.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn": module.runners_irsa.arn
    }
  }
}

resource "aws_iam_policy" "s3_bucket_access_policy" {
  name        = "s3-voxel-portal-${var.environment}"
  path        = "/"
  description = "the policy for allowing runner to read and write from s3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
        ]
        Effect   = "Allow"
        Resource = [
          "arn:aws:s3:::voxel-portal-${var.environment}",
          "arn:aws:s3:::voxel-portal-${var.environment}/*",
          "arn:aws:s3:::voxel-portal-${var.environment}-mumbai",
          "arn:aws:s3:::voxel-portal-${var.environment}-mumbai/*",
        ]
      },
    ]
  })
}


resource "aws_iam_policy" "voxel_storage_read_only" {
  name        = "s3-voxel-storage-read-only"
  path        = "/"
  description = "the policy for allowing runner to read models from voxel-storage"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
        ]
        Effect   = "Allow"
        Resource = [
          "arn:aws:s3:::voxel-storage/*",
        ]
      },
    ]
  })
}

module "runners_irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = "runners-production"
  serviceaccount = "runners-sa"
  oidc_url       = var.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.account_id}:oidc-provider/${var.oidc_provider}"
  policy_arns    = [
       "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess",
       resource.aws_iam_policy.s3_bucket_access_policy.arn,
       "arn:aws:iam::${var.account_id}:policy/PerceptionKinesisWriteOnly",
       resource.aws_iam_policy.voxel_storage_read_only.arn,
       aws_iam_policy.publish_to_prism_topic.arn,
       aws_iam_policy.access_polygon_graph_config_bucket.arn,
  ]
}

resource "aws_iam_role" "developer_access_assumable_role" {
  name = "developer_access_assumable_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = "AssumeRole"
        Principal = {
          AWS = [
            "arn:aws:iam::203670452561:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6",
          ]
        }
      },
    ]
  })
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess",
  ]
  max_session_duration = 43200
}

resource "aws_iam_role" "sematic_access_assumable_role" {
  name = "sematic_access_assumable_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Allow Sematic Worker SA in Prime Jenkins cluster to assume this role via IDP.
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Sid    = "AssumeRoleWithWebIdentity"
        Principal = {
          Federated = [
            "arn:aws:iam::${var.account_id}:oidc-provider/${local.prime_jenkins_eks_oidc_provider}"
          ]
        }
        Condition = {
          "StringEquals" = {
            "${local.prime_jenkins_eks_oidc_provider}:sub" = ["system:serviceaccount:sematic:sematic-worker"]
          }
        }
      },
    ]
  })
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonKinesisVideoStreamsReadOnlyAccess",
  ]
  max_session_duration = 43200
}

resource "aws_iam_policy" "publish_to_prism_topic" {
  name = "production_runner_publish_to_prism"
  policy = data.aws_iam_policy_document.publish_to_prism_topic.json
}

data "aws_iam_policy_document" "publish_to_prism_topic" {
  statement {
    sid = "PublishToPrism"
    actions = ["sns:Publish"]
    resources = ["arn:aws:sns:${var.region}:${var.account_id}:voxel-${var.environment}-prism-incident-ingest.fifo"]
  }
}

resource "aws_iam_policy" "access_polygon_graph_config_bucket" {
  name = "production_runner_access_polygon_graph_config_buket"
  policy = data.aws_iam_policy_document.access_polygon_graph_config_bucket.json
}

data "aws_iam_policy_document" "access_polygon_graph_config_bucket" {
  statement {
  actions = [
      "s3:Head*",
      "s3:Get*",
      "s3:Put*", 
    ]
    resources = ["arn:aws:s3:::voxel-${var.environment}-polygon-graph-configs"]
  }
}