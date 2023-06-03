resource "kubernetes_namespace" "flink_testing" {
  metadata {
    name = "flink-testing"
  }
}

resource "aws_iam_policy" "flink_testing_access" {
  name        = "FlinkTestingAccess"
  description = "Access for Flink Testing SA"
  policy = jsonencode({
    Statement = [
      {
        Sid      = "AssumeProductionAccountRole"
        Action   = ["sts:AssumeRole"],
        Effect   = "Allow",
        Resource = ["arn:aws:iam::360054435465:role/flink_testing_access_assumable_role"]
      },
    ]
    Version = "2012-10-17"
  })
}

resource "kubernetes_service_account" "flink_testing" {
  metadata {
    name      = "flink-testing"
    namespace = "flink-testing"
    annotations = {
      "eks.amazonaws.com/role-arn": module.flink_testing_irsa.arn
    }
  }
}

resource "kubernetes_role_binding" "flink_testing" {
  metadata {
    name      = "flink-testing"
    namespace = "flink-testing"
  }

  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Role"
    name      = "flink"
  }

  subject {
    kind      = "ServiceAccount"
    name      = "flink-testing"
    namespace = "flink-testing"
  }
}


module "flink_testing_irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = "flink-testing"
  serviceaccount = "flink-testing"
  oidc_url       = var.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.account_id}:oidc-provider/${var.oidc_provider}"
  policy_arns = [
    aws_iam_policy.flink_testing_access.arn,
  ]
}
