data "aws_iam_roles" "roles" {
  name_regex  = ".*AdministratorAccess.*"
  path_prefix = "/aws-reserved/sso.amazonaws.com/"
}

data "aws_iam_roles" "terraform_roles" {
  name_regex  = ".*TerraformAccess.*"
  path_prefix = "/"
}

locals {
  oidc_provider = replace(data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer, "https://", "")
  sso_role_arns = [
    for parts in [for arn in data.aws_iam_roles.roles.arns : split("/", arn)] :
    {
      rolearn  = format("%s/%s", parts[0], element(parts, length(parts) - 1))
      username = parts[length(parts) - 1]
      groups   = ["system:masters"]
    }
  ]
  terraform_role_arns = [
    for parts in [for arn in data.aws_iam_roles.terraform_roles.arns : split("/", arn)] :
    {
      rolearn  = format("%s/%s", parts[0], element(parts, length(parts) - 1))
      username = parts[length(parts) - 1]
      groups   = ["system:masters"]
    }
  ]
}

# module "eks_auth" {
#   source                      = "aidanmelen/eks-auth/aws"
#   version                     = "~>0.4.2"
#   eks_aws_auth_configmap_yaml = module.eks.aws_auth_configmap_yaml
#   map_roles = concat(
#     [
#       {
#         rolearn  = "arn:aws:iam::${var.account_id}:role/AdministratorAccess"
#         username = "administrator"
#         groups   = ["system:masters"]
#       }
#     ], concat(local.sso_role_arns, var.k8s_auth_extra_config)
#   )
# }

resource "kubernetes_namespace" "bindings_ns" {
  for_each = toset([for k, v in var.k8s_group_role_bindings : v.namespace])
  metadata {
    name = each.key
  }
}

resource "kubernetes_role_binding" "bindings" {
  for_each = var.k8s_group_role_bindings
  metadata {
    name      = each.key
    namespace = each.value.namespace # kubernetes_namespace.bindings_ns[each.key].metadata[0].name
  }
  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = each.value.role
  }
  subject {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Group"
    name      = each.value.group
  }
}

resource "kubernetes_cluster_role" "eks_console_dashboard_full_view_access_clusterrole" {
  metadata {
    name = "eks-console-dashboard-full-view-access-clusterrole"
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = [""]
    # Excluded secrets from here.
    resources  = ["nodes", "namespaces", "pods", "configmaps", "endpoints", "events", "limitranges", "persistentvolumeclaims", "podtemplates", "replicationcontrollers", "resourcequotas", "serviceaccounts", "services"]
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = ["apps"]
    resources  = ["deployments", "daemonsets", "statefulsets", "replicasets"]
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = ["batch"]
    resources  = ["jobs", "cronjobs"]
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = ["coordination.k8s.io"]
    resources  = ["leases"]
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = ["discovery.k8s.io"]
    resources  = ["endpointslices"]
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = ["events.k8s.io"]
    resources  = ["events"]
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = ["extensions"]
    resources  = ["daemonsets", "deployments", "ingresses", "networkpolicies", "replicasets"]
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = ["networking.k8s.io"]
    resources  = ["ingresses", "networkpolicies"]
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = ["policy"]
    resources  = ["poddisruptionbudgets"]
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = ["rbac.authorization.k8s.io"]
    resources  = ["rolebindings", "roles"]
  }

  rule {
    verbs      = ["get", "list"]
    api_groups = ["storage.k8s.io"]
    resources  = ["csistoragecapacities"]
  }
}

resource "kubernetes_cluster_role_binding" "eks_console_dashboard_full_view_access_binding" {
  metadata {
    name = "eks-console-dashboard-full-view-access-binding"
  }

  subject {
    kind = "Group"
    name = "eks-console-dashboard-full-view-access-group"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = "eks-console-dashboard-full-view-access-clusterrole"
  }
}