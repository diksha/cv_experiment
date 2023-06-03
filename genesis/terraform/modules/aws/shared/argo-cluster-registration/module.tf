resource "argocd_cluster" "cluster" {
  server     = data.aws_eks_cluster.cluster.endpoint
  name       = var.cluster_alias
  namespaces = []

  config {
    bearer_token = kubernetes_secret_v1.argocd_manager_secret.data.token
    tls_client_config {
      ca_data = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
    }
  }
}

resource "kubernetes_cluster_role" "argocd_manager" {
  metadata {
    name = "argocd-manager-role"
  }

  rule {
    api_groups = ["*"]
    resources  = ["*"]
    verbs      = ["*"]
  }

  rule {
    non_resource_urls = ["*"]
    verbs             = ["*"]
  }
}

resource "kubernetes_service_account" "argocd_manager" {
  metadata {
    name      = "argocd-manager"
    namespace = "kube-system"
  }
}

resource "kubernetes_secret_v1" "argocd_manager_secret" {
  metadata {
    namespace = "kube-system"
    name = "argocd-manager-sa-secret"
    annotations = {
      "kubernetes.io/service-account.name" = "argocd-manager"
    }
  }
  type = "kubernetes.io/service-account-token"
}


resource "kubernetes_cluster_role_binding" "argocd_manager" {
  metadata {
    name = "argocd-manager-role-binding"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = kubernetes_cluster_role.argocd_manager.metadata.0.name
  }

  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.argocd_manager.metadata.0.name
    namespace = kubernetes_service_account.argocd_manager.metadata.0.namespace
  }
}

data "kubernetes_secret" "argocd_manager" {
  metadata {
    name      = "argocd-manager-sa-secret"
    namespace = kubernetes_service_account.argocd_manager.metadata.0.namespace
  }
}
