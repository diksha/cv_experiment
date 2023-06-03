locals {
  name = var.name
  namespace = var.create_namespace ? kubernetes_namespace_v1.this[0].metadata[0].name : var.namespace
  service_account = kubernetes_service_account_v1.this.metadata[0].name
  oidc_provider = var.oidc_provider
}

data "aws_caller_identity" "this" {}

resource "kubernetes_namespace_v1" "this" {
  count = var.create_namespace ? 1 : 0
  metadata {
    name = var.namespace
  }
}

resource "kubernetes_service_account_v1" "this" {
  metadata {
    name      = local.name
    namespace = local.namespace
    annotations = {
      "eks.amazonaws.com/role-arn"     = module.this-irsa.arn
    }
  }
}

data "aws_iam_policy_document" "this" {
  statement {
    sid = "BucketReadOnly"
    actions = [
      "s3:ListBucket",
      "s3:ListObject",
      "s3:GetObject",
    ]
    resources = [
      "arn:aws:s3:::voxel-storage",
      "arn:aws:s3:::voxel-storage/*",
      "arn:aws:s3:::voxel-production-triton-models",
      "arn:aws:s3:::voxel-production-triton-models/*",
      "arn:aws:s3:::voxel-staging-triton-models",
      "arn:aws:s3:::voxel-staging-triton-models/*",
      "arn:aws:s3:::voxel-production-triton-models",
      "arn:aws:s3:::voxel-production-triton-models/*",
      "arn:aws:s3:::voxel-experimental-triton-models",
      "arn:aws:s3:::voxel-experimental-triton-models/*",
    ]
  }
}

resource "aws_iam_policy" "this" {
  name = "TritonExperimentServiceAccessPolicy-${var.name}"
  policy = data.aws_iam_policy_document.this.json
}

module "this-irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = local.namespace
  // we can't use a service account reference here since the service account depends on this for an annotation
  serviceaccount = local.name
  oidc_url       = local.oidc_provider
  oidc_arn       = "arn:aws:iam::${data.aws_caller_identity.this.account_id}:oidc-provider/${local.oidc_provider}"
  policy_arns    = [aws_iam_policy.this.arn]
}

resource "kubernetes_deployment_v1" "this" {
  metadata {
    name = var.name
    namespace = local.namespace
    labels = {
      app = var.name
    }
  }
  spec {
    replicas = 1
    selector {
      match_labels = {
        app = var.name
      }
    }

    template {
      metadata {
        name = local.name
        namespace = local.namespace
        labels = {
          app = var.name
        }
      }

      spec {
        node_selector = {
          "node.kubernetes.io/instance-type" = "g4dn.xlarge"
        }

        service_account_name = local.service_account
        container {
          name = var.name
          image = "nvcr.io/nvidia/tritonserver:22.07-py3"
          command = ["tritonserver"]
          port {
            name = "grpc-requests"
            container_port = 8001
          }
          port {
              name = "metrics"
              container_port = 8002
          }
          args = [
            "--model-repository=${var.model_repository}",
            "--model-control-mode=${var.model_control_mode}",
            "--disable-auto-complete-config",
          ]
          resources {
            limits = {
              cpu = "3500m"
              memory = "14Gi"
            }
            requests = {
              cpu = "3500m"
              memory = "14Gi"
            }
          }
        }
        toleration {
          effect = "NoSchedule"
          key = "nvidia.com/gpu"
          value = "true"
        }
      }
    }
  }
}

resource "kubernetes_service_v1" "this" {
  metadata {
    name = var.name
    namespace = local.namespace
    annotations = {
      "service.beta.kubernetes.io/aws-load-balancer-type" = "external"
      "service.beta.kubernetes.io/aws-load-balancer-nlb-target-type" = "ip"
      "service.beta.kubernetes.io/aws-load-balancer-additional-resource-tags" = "map-migrated=d-server-00swbp99drezfh"
    }
  }
  spec {
    selector = {
      app = var.name
    }
    port {
      name        = "grpc-requests"
      port        = 8001
      target_port = 8001
    }
    port {
      name        = "metrics"
      port        = 8002
      target_port = 8002
    }
    type = "LoadBalancer"
  }
}