resource "kubernetes_ingress_v1" "alert_manager_ingress" {
  metadata {
    name      = "alert-manager-ingress"
    namespace = "observability"
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "alert-manager.${var.observability_identifier}-observability.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "observability-stack-kube-p-alertmanager"
              port {
                number = 9093
              }
            }
          }
          path = "/"
        }
      }
    }
  }
}


resource "grafana_data_source" "alertmanager" {
  count               = var.register_with_grafana ? 1 : 0
  type                = "alertmanager"
  name                = "alertmanager-${var.observability_identifier}"
  # url                 = "http://${data.kubernetes_service.alertmanager.status[0].load_balancer[0].ingress[0].hostname}:9093"
  url                 = "http://alert-manager.${var.observability_identifier}-observability.voxelplatform.com"
  basic_auth_enabled  = false
  basic_auth_username = local.prometheus_username
  is_default          = false
  secure_json_data {
    basic_auth_password = random_password.prometheus_password.result
  }
  json_data {
    implementation = "prometheus"
  }
}


resource "kubernetes_ingress_v1" "tempo_ingress" {
  metadata {
    name      = "tempo-ingress"
    namespace = "observability"
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "tempo.${var.observability_identifier}-observability.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "tempo-tempo-distributed-gateway"
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


resource "grafana_data_source" "tempo" {
  count               = var.register_with_grafana ? 1 : 0
  type                = "tempo"
  name                = "tempo-${var.observability_identifier}"
  # url                 = "http://${data.kubernetes_service.tempo.status[0].load_balancer[0].ingress[0].hostname}"
  url                 = "http://tempo.${var.observability_identifier}-observability.voxelplatform.com"
  basic_auth_enabled  = false
  basic_auth_username = local.tempo_username
  is_default          = false
  secure_json_data {
    basic_auth_password = random_password.tempo_password.result
  }
}


resource "kubernetes_ingress_v1" "prometheus_ingress" {
  metadata {
    name      = "prometheus-ingress"
    namespace = "observability"
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "prometheus.${var.observability_identifier}-observability.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "observability-stack-kube-p-prometheus"
              port {
                number = 9090
              }
            }
          }
          path = "/"
        }
      }
    }
  }
}


resource "grafana_data_source" "prometheus" {
  count               = var.register_with_grafana ? 1 : 0
  type                = "prometheus"
  name                = "prometheus-${var.observability_identifier}"
  # url                 = "http://${data.kubernetes_service.prometheus.status[0].load_balancer[0].ingress[0].hostname}:9090"
  url                 = "http://prometheus.${var.observability_identifier}-observability.voxelplatform.com"
  basic_auth_enabled  = true
  basic_auth_username = local.prometheus_username
  secure_json_data {
    basic_auth_password = random_password.prometheus_password.result
  }
  json_data {
    http_method = "POST"
  }
}

resource "kubernetes_ingress_v1" "loki_ingress" {
  metadata {
    name      = "loki-ingress"
    namespace = "observability"
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "loki.${var.observability_identifier}-observability.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "loki-distributed-gateway"
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


resource "grafana_data_source" "loki" {
  count               = var.register_with_grafana ? 1 : 0
  type                = "loki"
  name                = "loki-${var.observability_identifier}"
  # url                 = "http://${data.kubernetes_service.loki.status[0].load_balancer[0].ingress[0].hostname}"
  url                 = "http://loki.${var.observability_identifier}-observability.voxelplatform.com"
  basic_auth_enabled  = true
  basic_auth_username = local.loki_username
  is_default          = false
  json_data {
    derived_field {
      datasource_uid = grafana_data_source.tempo[0].uid
      matcher_regex  = "traceid=(\\w+)"
      name           = "traceID"
      url            = "$${__value.raw}"
    }
  }
  secure_json_data {
    basic_auth_password = random_password.loki_password.result
  }
}

resource "grafana_data_source" "cloudwatch" {
  count = var.register_with_grafana ? 1 : 0
  type  = "cloudwatch"
  name  = "cloudwatch-${var.observability_identifier}"
  json_data {
    default_region  = var.aws_region
    assume_role_arn = aws_iam_role.observability_delegate[0].arn
    auth_type       = "default"
  }
}

resource "grafana_data_source" "cloudwatch_alarm" {
  count = var.register_with_grafana ? 1 : 0
  type  = "computest-cloudwatchalarm-datasource"
  name  = "cloudwatch-alarm-${var.observability_identifier}"
  json_data {
    default_region  = var.aws_region
    assume_role_arn = aws_iam_role.observability_delegate[0].arn
    auth_type       = "default"
  }
}


resource "aws_iam_role" "observability_delegate" {
  count               = var.register_with_grafana ? 1 : 0
  name                = "cloudwatch-delegate-${var.observability_identifier}-access"
  managed_policy_arns = ["arn:aws:iam::aws:policy/CloudWatchReadOnlyAccess"]
  assume_role_policy  = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
      {
          "Effect": "Allow",
          "Action": "sts:AssumeRole",
          "Principal": {
              "AWS": "${var.grafana_irsa_arn}"
          },
          "Condition": {}
      },
      {
          "Effect": "Allow",
          "Action": "sts:AssumeRoleWithWebIdentity",
          "Principal": {
              "AWS": "${var.grafana_irsa_arn}"
          },
          "Condition": {}
      }
  ]
}
EOF
}
