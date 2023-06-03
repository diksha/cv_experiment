locals {
  service_name                               = var.name
  full_name                                  = "${local.service_name}-service"
  argo_project_name                          = "${local.service_name}-${local.environment}"
  secret_provider_class_name                 = "${local.full_name}-secret-provider-class"
  argo_cluster_name                          = "${var.context.environment}-${var.context.eks_cluster_name}"
  environment                                = var.context.environment
}

resource "aws_ecr_repository" "service" {
  name = "${local.environment}/service/${local.service_name}"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

data "aws_iam_policy_document" "repo" {
  statement {
    sid = "AllowBuildkitePush"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:PutImage",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload"
    ]
    principals {
      type = "AWS"
      identifiers = ["arn:aws:iam::203670452561:role/BuildkiteAccess"]
    }
  }
}

resource "aws_ecr_repository_policy" "repo" {
  repository = aws_ecr_repository.service.name
  policy = data.aws_iam_policy_document.repo.json
} 

data "aws_iam_policy_document" "secret_access" {
  statement {
    sid = "GetServiceSecret"
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret",
    ]
    resources = [
      aws_secretsmanager_secret.service.arn,
    ]
  }
}

resource "aws_iam_policy" "secret_access" {
  name        = "${local.environment}-${local.full_name}-secret-access"
  path        = "/"
  description = "the policy for allowing pods to read from secret manager"

  policy = data.aws_iam_policy_document.secret_access.json
}

module "irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = kubernetes_namespace_v1.service.metadata[0].name
  // we can't use a service account reference here since the service account depends on this for an annotation
  serviceaccount = local.full_name
  oidc_url       = var.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.context.target_account_id}:oidc-provider/${var.oidc_provider}"
  policy_arns = concat([
    resource.aws_iam_policy.secret_access.arn,
  ], var.extra_policy_arns)
}

resource "kubernetes_namespace_v1" "service" {
  metadata {
    name = local.full_name
    labels = {
      "autocert.step.sm" = "enabled"
    }
  }
}

resource "kubernetes_service_account_v1" "service" {
  metadata {
    name      = local.full_name
    namespace = kubernetes_namespace_v1.service.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" : module.irsa.arn
      "meta.helm.sh/release-name"      = local.secret_provider_class_name
      "meta.helm.sh/release-namespace" = kubernetes_namespace_v1.service.metadata[0].name
    }
    labels = {
      "app.kubernetes.io/instance"   = local.secret_provider_class_name
      "app.kubernetes.io/managed-by" = "Helm"
      "app.kubernetes.io/name"       = "SecretProviderClass"
      "app.kubernetes.io/version"    = "1.16.0"
      "helm.sh/chart"                = "SecretProviderClass-0.1.0"
    }

  }
}

resource "aws_secretsmanager_secret" "service" {
  name = "service/${local.service_name}"
  description = "Secrets for ${local.full_name}"
}


resource "helm_release" "SecretProviderClass" {
  depends_on = [
    module.irsa
  ]
  count            = 1
  atomic           = true
  cleanup_on_fail  = true
  name             = local.secret_provider_class_name
  namespace        = kubernetes_namespace_v1.service.metadata[0].name
  chart            = "${path.module}/../../../shared/charts/SecretProviderClass"
  version          = "0.1.0"
  wait             = false
  timeout          = 300
  replace          = false
  create_namespace = false
  force_update     = true
  values = [templatefile("${path.module}/files/generic-service-secret-provider-class-values.yaml",
    {
      IAM_ROLE_ARN         = module.irsa.arn
      NAME                 = local.full_name
      SM_NAME              = aws_secretsmanager_secret.service.name
      SERVICE_ACCOUNT_NAME = kubernetes_service_account_v1.service.metadata[0].name
      SP_CLASS_NAME        = local.secret_provider_class_name
      ENVIRONMENT          = local.environment
    })
  ]
}

resource "argocd_project" "service" {
  depends_on = [
    module.irsa
  ]
  metadata {
    name      = local.argo_project_name
    namespace = "argo-cd"
    labels = {
      acceptance = "true"
    }
  }
  spec {
    description  = "${local.service_name} service"
    source_repos = ["git@github.com:voxel-ai/genesis"]

    destination {
      name = local.argo_cluster_name
      namespace = kubernetes_namespace_v1.service.metadata[0].name
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
        "p, proj:${local.argo_project_name}:view, applications, get, ${local.argo_project_name}/*, allow",
      ]
      groups = [
        "engineering@voxelai.com"
      ]
    }
  }
}

resource "argocd_application" "service" {
  depends_on = [
    module.irsa
  ]
  metadata {
    name      = argocd_project.service.metadata[0].name
    namespace = "argo-cd"
    annotations = {
      "notifications.argoproj.io/subscribe.on-created.slack"   = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-health-degraded" = "devops-alerts"
      "notifications.argoproj.io/subscribe.on-sync-failed"     = "devops-alerts"
    }
  }

  wait = true
  spec {
    project = local.argo_project_name
    source {
      path            = "terraform/modules/aws/shared/charts/generic"
      repo_url        = "git@github.com:voxel-ai/genesis"
      target_revision = "main"
      helm {
        parameter {
          name         = "image.tag"
          value        = var.initial_image_tag
          force_string = false
        }

        # value_files  = ["values-test.yml"]
        values = templatefile("${path.module}/files/generic-service-values.yaml",
          {
            NAME                 = local.service_name
            FULL_NAME            = local.full_name
            SERVICE_ACCOUNT_NAME = kubernetes_service_account_v1.service.metadata[0].name
            IMAGE_REPO           = aws_ecr_repository.service.repository_url
            SECRET_PROVIDER_NAME = local.secret_provider_class_name
            SM_NAME              = aws_secretsmanager_secret.service.name
            ENVIRONMENT          = local.environment
            IMAGE_TAG            = var.initial_image_tag
            TLS_COMMON_NAME      = "${local.full_name}.${kubernetes_namespace_v1.service.metadata[0].name}.svc.cluster.local"
            TLS_SANS             = join(",", var.tls_sans)
        })

        release_name = local.full_name
      }
    }

    destination {
      name = local.argo_cluster_name
      namespace = kubernetes_namespace_v1.service.metadata[0].name
    }

    sync_policy {
      automated = {
        allow_empty = false
        prune       = true
        self_heal   = true
      }
      sync_options = ["CreateNamespace=false", "Replace=true"]
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
  lifecycle {
    ignore_changes = [
      spec[0].source[0].helm[0].parameter
    ]
  }

}

