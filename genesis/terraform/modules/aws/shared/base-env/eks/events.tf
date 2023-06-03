resource "helm_release" "kubernetes_event_exporter" {
  name             = "kubernetes-event-exporter"
  namespace        = "kube-system"
  create_namespace = true
  repository       = "https://charts.bitnami.com/bitnami"
  chart            = "kubernetes-event-exporter"
  version          = "v2.1.5"
  values = [
    jsonencode({
      config = {
        logLevel           = "debug"
        logFormat          = "json"
        maxEventAgeSeconds = 60
        receivers = [
          {
            name   = "dump"
            stdout = { }
          }
        ]
      }
      }
  )]
}
