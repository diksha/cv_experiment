resource "aws_acm_certificate" "cert" {
  domain_name       = "*.private.voxelplatform.com"
  validation_method = "DNS"
  subject_alternative_names = var.subject_alternative_names
  lifecycle {
    create_before_destroy = true
  }
}


# TODO: Add support for https://kubernetes-sigs.github.io/aws-load-balancer-controller/v2.0/guide/controller/pod_readiness_gate/
resource "helm_release" "main" {
  count            = var.enable_nginx_ingress ? 1 : 0
  name             = "ingress-nginx"
  namespace        = "ingress-nginx"
  chart            = "ingress-nginx"
  repository       = "https://kubernetes.github.io/ingress-nginx"
  version          = "4.4.0"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
      STATIC_IPv4s = var.static_ipv4s
      SUBNET_IDS      = var.subnet_ids
      CERTIFICATE_ARN = aws_acm_certificate.cert.arn
      SSL_PORTS       = var.ssl_ports
    })
  ]
  dynamic "set" {
    for_each = var.tcp_services
    content {
      name = "tcp.${set.key}"
      value  = set.value
    }
  }
}

# resource "kubernetes_labels" "pod_readiness_gate_label" {
#   api_version = "v1"
#   kind        = "Namespace"
#   metadata {
#     name = "ingress-nginx"
#     namespace = "ingress-nginx"
#   }
#   labels = {
#     "elbv2.k8s.aws/pod-readiness-gate-inject" = "enabled"
#   }
# }