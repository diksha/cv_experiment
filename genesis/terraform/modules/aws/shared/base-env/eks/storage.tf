module "ebs_irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = "storage"
  serviceaccount = "ebs-csi-controller-sa"
  oidc_url       = module.eks.oidc_provider
  oidc_arn       = module.eks.oidc_provider_arn
  policy_arns    = ["arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"]
}

resource "helm_release" "ebs_csi" {
  name             = "ebs-csi"
  repository       = "https://kubernetes-sigs.github.io/aws-ebs-csi-driver"
  chart            = "aws-ebs-csi-driver"
  version          = "2.7.0"
  namespace        = "storage"
  create_namespace = true
  values = [
    jsonencode({
      node = {
        tolerateAllTaints = true
        serviceAccount = {
          annotations = {
            "eks.amazonaws.com/role-arn" = module.ebs_irsa.arn
          }
        }
      }
      controller = {
        serviceAccount = {
          annotations = {
            "eks.amazonaws.com/role-arn" = module.ebs_irsa.arn
          }
        }
      }
      storageClasses = [
        {
          name          = "gp3-retain"
          reclaimPolicy = "Retain"
          parameters = {
            type      = "gp3"
            fsType    = "ext4"
            encrypted = "true"
          }
          allowVolumeExpansion = true
        },
        {
          name          = "gp3"
          reclaimPolicy = "Delete"
          parameters = {
            type      = "gp3"
            fsType    = "ext4"
            encrypted = "true"
          }
          allowVolumeExpansion = true
        }
      ]
  })]
}

resource "kubernetes_storage_class" "gp2_retain" {
  metadata {
    name = "gp2-retain"

  }
  storage_provisioner = "kubernetes.io/aws-ebs"
  reclaim_policy      = "Retain"
  parameters = {
    type      = "gp2"
    fsType    = "ext4"
    encrypted = "true"
  }
  allow_volume_expansion = true
  volume_binding_mode    = "Immediate"
}
