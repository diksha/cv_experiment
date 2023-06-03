locals {
  prometheus_username = "prometheus"
  loki_username       = "loki"
  tempo_username      = "tempo"
  oidc_provider       = replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")
}

resource "random_password" "prometheus_password" {
  length           = 20
  special          = true
  override_special = "_-"
}


resource "random_password" "loki_password" {
  length           = 20
  special          = true
  override_special = "_"
}

resource "random_password" "tempo_password" {
  length           = 20
  special          = true
  override_special = "_-"
}

resource "kubernetes_service_account" "observability_aws" {
  metadata {
    name      = local.aws_service_account_name
    namespace = "observability"
    annotations = {
      "eks.amazonaws.com/role-arn" = module.observability_irsa.arn
    }
  }
  depends_on = [
    helm_release.observability_stack
  ]
}

resource "helm_release" "observability_stack" {
  name             = "observability-stack"
  namespace        = "observability"
  chart            = "kube-prometheus-stack"
  repository       = "https://prometheus-community.github.io/helm-charts"
  version          = "35.5.1"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [
    templatefile("${path.module}/ymls/kube-prometheus-stack-helm-values.yml", {
      FORCE_DEPLOY_DASHBOARD = var.force_deploy_dashboard_config_maps
      PROMETHEUS_MEMORY = var.prometheus_memory
    })
  ]
  set_sensitive {
    name  = "prometheus.extraSecret.auth"
    value = "${local.prometheus_username}:${random_password.prometheus_password.bcrypt_hash}"
  }

  set_sensitive {
    name  = "alertmanager.extraSecret.auth"
    value = "${local.prometheus_username}:${random_password.prometheus_password.bcrypt_hash}"
  }
}

resource "helm_release" "loki_distributed" {
  name             = "loki-distributed"
  namespace        = "observability"
  chart            = "loki-distributed"
  repository       = "https://grafana.github.io/helm-charts"
  version          = "0.48.4"
  create_namespace = true
  wait             = true
  timeout          = 150
  values = [
    templatefile("${path.module}/ymls/loki-distributed.yml", {
      AWS_REGION           = var.aws_region
      BUCKET_NAME          = module.loki_bucket.bucket_id
      USERNAME             = local.loki_username
      SERVICE_ACCOUNT_NAME = kubernetes_service_account.observability_aws.metadata[0].name
    })
  ]

  set_sensitive {
    name  = "gateway.basicAuth.password"
    value = random_password.loki_password.result
  }
}


resource "helm_release" "tempo" {
  name             = "tempo"
  namespace        = "observability"
  create_namespace = true
  repository       = "https://grafana.github.io/helm-charts"
  chart            = "tempo-distributed"
  version          = "0.19.3"
  wait             = true
  timeout          = 300
  values = [
    templatefile("${path.module}/ymls/tempo-distributed.yml", {
      AWS_REGION           = var.aws_region
      BUCKET_NAME          = module.tempo_bucket.bucket_id
      USERNAME             = local.tempo_username
      SERVICE_ACCOUNT_NAME = kubernetes_service_account.observability_aws.metadata[0].name
    })
  ]
  set_sensitive {
    name  = "gateway.basicAuth.password"
    value = random_password.tempo_password.result
  }
  set_sensitive {
    name  = "metricsGenerator.config.storage_remote_write[0].basic_auth.username"
    value = local.prometheus_username
  }

  set_sensitive {
    name  = "metricsGenerator.config.storage_remote_write[0].basic_auth.password"
    value = random_password.prometheus_password.result
  }
}

resource "helm_release" "nvidia_dcgm_exporter" {
  count            = var.install_gpu_components ? 1 : 0
  name             = "nvidia-dcgm-exporter"
  namespace        = "observability"
  create_namespace = true
  repository       = "https://nvidia.github.io/dcgm-exporter/helm-charts"
  chart            = "dcgm-exporter"
  version          = "3.0.0"
  timeout          = 1200
  values = [
    templatefile("${path.module}/ymls/nvidia-dcgm-exporter.yml", {
    })
  ]
  depends_on = [helm_release.observability_stack]
}

resource "helm_release" "fluent_bit" {
  name             = "fluent-bit"
  namespace        = "observability"
  create_namespace = true
  repository       = "https://fluent.github.io/helm-charts"
  chart            = "fluent-bit"
  version          = "0.23.0"
  timeout          = 1200
  values = [
    templatefile("${path.module}/ymls/fluent-bit.yml", {
    })
  ]
  depends_on = [helm_release.observability_stack]
}