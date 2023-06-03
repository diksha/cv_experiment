resource "helm_release" "gpu_manifest" {
  name             = "gpu-manifests"
  chart            = "https://github.com/itscontained/charts/releases/download/raw-v0.2.5/raw-v0.2.5.tgz"
  version          = "v0.2.5"
  namespace        = "gpu-manifests"
  create_namespace = true
  values           = [file("${path.module}/yamls/nvidia-device-plugin.yml")]
}

module "alb_ingress_controller" {
  source                    = "git::https://github.com/getmiso/terraform-kubernetes-aws-load-balancer-controller.git?ref=579625f"
  k8s_cluster_type          = "eks"
  k8s_namespace             = "kube-system"
  k8s_cluster_name          = module.eks.cluster_name
  aws_load_balancer_controller_chart_version = "1.4.6"
  enable_host_networking = true
  alb_controller_depends_on = [] 
}

# resource "kubernetes_secret" "alb_ingress_controller_sa_token" {
#   metadata {
#     name        = "aws-load-balancer-controller-token"
#     namespace   = "kube-system"
#     annotations = {
#       "kubernetes.io/service-account.name" = "aws-load-balancer-controller"
#     }
#   }
#   type = "kubernetes.io/service-account-token"
# }


resource "helm_release" "cert_manager" {
  name             = "cert-manager"
  namespace        = "cert-manager"
  create_namespace = true
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  version          = "v1.5.5"
  values = [
    yamlencode({
      installCRDs = true
    })
  ]
}
