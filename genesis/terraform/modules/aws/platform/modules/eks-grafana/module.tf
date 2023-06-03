
resource "random_password" "admin_password" {
  length           = 20
  special          = true
  override_special = "_-"
}

resource "helm_release" "grafana" {
  name             = "grafana"
  namespace        = "observability"
  chart            = "grafana"
  repository       = "https://grafana.github.io/helm-charts"
  version          = "6.44.4"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [
    templatefile("${path.module}/files/values.yaml",
      {
        GOOGLE_CLIENT_ID     = var.google_client_id
        GOOGLE_CLIENT_SECRET = var.google_client_secret
        DOMAIN               = var.domain
        ADMIN_PASSWORD       = random_password.admin_password.result
        SLACK_TOKEN          = var.slack_token
        IAM_ROLE_ARN         = module.grafana_irsa.arn
    })
  ]
}

module "grafana_irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = "observability"
  serviceaccount = "grafana"
  oidc_url       = var.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.account_id}:oidc-provider/${var.oidc_provider}"
  policy_arns    = []
}

resource "aws_iam_role_policy" "assume_role_for_irsa" {
  name = "assume-role"
  role = module.grafana_irsa.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "sts:AssumeRole",
        ]
        Effect   = "Allow"
        Resource = "*"
      },
    ]
  })
}
