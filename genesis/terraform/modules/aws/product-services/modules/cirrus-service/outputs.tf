output "namespace" {
    value = kubernetes_namespace_v1.service.metadata[0].name
}