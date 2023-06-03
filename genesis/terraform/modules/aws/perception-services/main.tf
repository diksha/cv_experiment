locals {
  prime_jenkins_eks_oidc_provider = "oidc.eks.us-west-2.amazonaws.com/id/7F736585AF1A879653C8E0819543D12E"
  oidc_provider = replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")
}

module "perception_services_setup" {
  providers = {
    aws = aws
  }
  source                                        = "./base-services"
  account_id                                    = var.target_account_id
  google_application_credentials_base64_encoded = var.google_application_credentials_base64_encoded
  sentry_dsn                                    = var.sentry_dsn
  cluster_name                                  = var.eks_cluster_name
  environment                                   = var.environment
  oidc_provider                                 = local.oidc_provider
  region                                        = var.primary_region
}

resource "aws_ecr_repository" "fisheye" {
  name = "${var.eks_cluster_name}/fisheye"
  image_tag_mutability = "IMMUTABLE"

}

resource "aws_ecr_lifecycle_policy" "foopolicy" {
  repository = aws_ecr_repository.fisheye.name
  policy = <<EOF
{
    "rules": [
        {
            "rulePriority": 1,
            "description": "Keep last 5 images",
            "selection": {
                "tagStatus": "any",
                "countType": "imageCountMoreThan",
                "countNumber": 5
            },
            "action": {
                "type": "expire"
            }
        }
    ]
}
EOF
}

resource "argocd_project" "perception_fisheye_project" {
  metadata {
    name      = "perception-fisheye-${var.environment}"
    namespace = "argo-cd"
    labels = {
      acceptance = "true"
    }
  }
  spec {
    description  = "simple project"
    source_repos = ["git@github.com:voxel-ai/genesis"]

    destination {
      server    = "*"
      namespace = "*"
      name      = "*"
    }
    cluster_resource_whitelist {
      group = "*"
      kind  = "*"
    }
    namespace_resource_whitelist {
      group = "*"
      kind  = "*"
    }
    role {
      name = "view"
      policies = [
        "p, proj:perception-fisheye-${var.environment}:view, applications, get, perception-fisheye-${var.environment}/*, allow",
      ]
      groups = [
        "engineering@voxelai.com"
      ]
    }
  }
}


resource "argocd_application" "perception_fisheye" {
  metadata {
    name      = "perception-fisheye"
    namespace = "argo-cd"
    annotations = {
      "notifications.argoproj.io/subscribe.on-created.slack"   = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-health-degraded" = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-sync-failed"     = "devops-alerts"
    }
  }

  wait = true
  spec {
    project = "perception-fisheye-${var.environment}"
    source {
      path            = "terraform/modules/aws/shared/charts/generic"
      repo_url        = "git@github.com:voxel-ai/genesis"
      target_revision = "main"
      helm {
        values = templatefile("${path.module}/files/fisheye-values.yaml",
          {
            IMAGE_REPO           = aws_ecr_repository.fisheye.repository_url
            IMAGE_TAG            = "vai_1668815122"
            ENVIRONMENT          = var.environment
        })

        release_name = "perception-fisheye"
      }
    }

    destination {
      name      = "${var.environment}-perception"
      namespace = "perception-fisheye-${var.environment}"
    }
    sync_policy {
      automated = {
        prune       = true
        self_heal   = true
        allow_empty = false
      }
      sync_options = ["CreateNamespace=true", "Replace=true"]
      retry {
        limit = "5"
        backoff = {
          duration     = "30s"
          max_duration = "2m"
          factor       = "2"
        }
      }
    }
  }
}

resource "kubernetes_ingress_v1" "fisheye_ingress" {
  metadata {
    name      = "fisheye-ingress"
    namespace = "perception-fisheye-${var.environment}"
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "fisheye.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "perception-fisheye-generic"
              port {
                number = 80
              }
            }
          }
          path = "/"
        }
      }
    }
  }
}

resource "aws_iam_role" "flink_testing_access_assumable_role" {
  name = "flink_testing_access_assumable_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Allow Flink Testing SA in Prime Jenkins cluster to assume this role via IDP.
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = "AssumeRoleFromPrimeDevops"
        Principal = {
          AWS = [
            "arn:aws:iam::203670452561:role/irsa-40gd1t55uqug",
          ]
        }
      },
    ]
  })
  managed_policy_arns  = [aws_iam_policy.flink_testing_access.arn]
  max_session_duration = 43200
}

resource "aws_iam_policy" "flink_testing_access" {
  name = "FlinkTestingAccess"
  description = "Flink Testing access permissions"
  policy = data.aws_iam_policy_document.flink_testing_access.json
}

data "aws_iam_policy_document" "flink_testing_access" {
  statement {
    sid = "StatesEventsS3Read"
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
      "s3:GetBucketLocation"
    ]
    resources = [
      var.states_events_bucket_arn,
      "${var.states_events_bucket_arn}/*",
    ]
  }
}

module "triton_model_repos" {
  source            = "../shared/s3-bucket"
  target_account_id = var.target_account_id
  primary_region    = var.primary_region
  bucket_name       = "voxel-${var.environment}-triton-models"
  enable_versioning = true
  noncurrent_days   = 30
}

resource "aws_s3_bucket_policy" "triton_model_repos" {
  bucket = module.triton_model_repos.bucket_id
  policy = data.aws_iam_policy_document.triton_model_repos_access.json
}

data "aws_iam_policy_document" "triton_model_repos_access" {
  statement {
    sid = "readwrite"
    actions = [
      "s3:ListBucketMultipartUploads",
      "s3:ListBucketVersions",
      "s3:ListBucket",
      "s3:ListMultipartUploadParts",
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:PutObject",
      "s3:RestoreObject",
      "s3:AbortMultipartUpload",
      "s3:DeleteObject",
      "s3:GetBucketVersioning",
      "s3:GetBucketTagging",
    ]
    principals {
      type = "AWS"
      identifiers = ["arn:aws:iam::203670452561:role/aws-reserved/sso.amazonaws.com/us-west-2/AWSReservedSSO_DeveloperAccess_f74b7dd970403af6"]
    }
    resources = [
      module.triton_model_repos.bucket_arn,
      "${module.triton_model_repos.bucket_arn}/*"
    ]
  }
}
