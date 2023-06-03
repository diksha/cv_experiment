locals {
  platform_primary_vpc_cidr = "10.1.0.0/16"
}


module "sematic_db" {
  source                                  = "../../../shared/rds-postgres"
  account_id                              = var.account_id
  db_identifier                           = var.deplomyment_identifier
  subnet_ids                              = var.private_subnet_ids
  db_instance_size                        = "db.t4g.small"
  environment                             = var.environment
  vpc_id                                  = var.vpc_id
  ingress_cidr_blocks                     = concat(var.private_subnet_cidrs, [local.platform_primary_vpc_cidr])
  multi_az                                = false
  backup_replica                          = false
  create_rds_alarms_sns_and_slack         = false
  aws_notifications_slack_app_webhook_url = ""
}


resource "helm_release" "primary" {
  name             = "${var.deplomyment_identifier}-server"
  namespace        = var.deplomyment_identifier
  chart            = "sematic-server"
  repository       = "https://sematic-ai.github.io/helm-charts"
  wait             = true
  timeout          = 300
  replace          = false
  # Namespace created by Postgres module above.
  create_namespace = false
  version          = "1.1.9"
  values = [
    templatefile("${path.module}/files/service-helm-values.yaml",
      {
        GOOGLE_CLIENT_ID = var.google_client_id
    })
  ]
  set_sensitive {
    name  = "database.url"
    value = "postgresql://${module.sematic_db.db_instance_username}:${module.sematic_db.db_instance_password}@${module.sematic_db.db_instance_address}:${module.sematic_db.db_instance_port}/${module.sematic_db.db_instance_name}"
  }
  set_sensitive {
    name  = "slack.slack_webhook_token"
    value = var.perception_verbose_slack_hook_url_token
  }
}

resource "kubernetes_service_account" "sematic_worker" {
  metadata {
    name      = "${var.deplomyment_identifier}-worker"
    namespace = var.deplomyment_identifier
    annotations = {
      "eks.amazonaws.com/role-arn": module.sematic_worker_irsa.arn
    }
  }
}

module "sematic_worker_irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = var.deplomyment_identifier
  serviceaccount = "sematic-worker"
  oidc_url       = var.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.account_id}:oidc-provider/${var.oidc_provider}"
  policy_arns = [
    # Pass the policy as a variable to the module.
    "arn:aws:iam::${var.account_id}:policy/SematicAccess",
  ]
}

resource "aws_iam_policy" "sematic_server" {
  name        = "SematicServerAccess"
  description = "Access for Sematic Service"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid = "Buckets"
        Action = [
          "s3:ListBucketMultipartUploads",
          "s3:ListBucketVersions",
          "s3:ListBucket",
          "s3:ListMultipartUploadParts",
          "s3:GetObjectAcl",
          "s3:GetObject",
          "s3:GetBucketAcl",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersion",
          "s3:GetStorageLensDashboard",
          "s3:PutObject",
          "s3:RestoreObject",
          "s3:AbortMultipartUpload",
          "s3:DeleteObject",
          "s3:GetBucketVersioning",
          "s3:GetLifecycleConfiguration",
          "s3:GetEncryptionConfiguration",
          "s3:GetBucketTagging",
          "s3:GetBucketPolicy",
          "s3:GetBucketAcl",
        ],
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::${var.account_id}-sematic-ci-cd",
          "arn:aws:s3:::${var.account_id}-sematic-ci-cd/*",
        ]
  }]})
}

resource "kubernetes_service_account" "sematic_server" {
  metadata {
    name      = "${var.deplomyment_identifier}-server"
    namespace = var.deplomyment_identifier
    annotations = {
      "eks.amazonaws.com/role-arn": module.sematic_server_irsa.arn
    }
  }
}

module "sematic_server_irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  namespace      = var.deplomyment_identifier
  serviceaccount = "sematic-server"
  oidc_url       = var.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.account_id}:oidc-provider/${var.oidc_provider}"
  policy_arns = [
    aws_iam_policy.sematic_server.arn,
  ]
}

resource "helm_release" "kuberay_operator" {
  name             = "kuberay-operator"
  namespace        = var.deplomyment_identifier
  chart            = "kuberay-operator"
  repository       = "https://ray-project.github.io/kuberay-helm/"
  wait             = true
  timeout          = 300
  replace          = false
  create_namespace = false
  version          = "0.4.0"
  values = [
    templatefile("${path.module}/files/kuberay-operator-helm-values.yaml",
      {
    })
  ]
}

resource "grafana_data_source" "sematic_prime_postgres" {
  type          = "postgres"
  name          = "${var.deplomyment_identifier}-prime-postgres"
  url           = "${module.sematic_db.db_instance_address}:${module.sematic_db.db_instance_port}"
  database_name = module.sematic_db.db_instance_name
  username      = module.sematic_db.db_instance_username

  is_default = false
  json_data {
    postgres_version         = 1400
    ssl_mode                 = "disable"
  }

  secure_json_data {
    password = module.sematic_db.db_instance_password
  }
}
