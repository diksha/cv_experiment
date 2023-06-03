resource "helm_release" "release" {
  name             = "vault"
  namespace        = "vault"
  chart            = "vault"
  repository       = "https://helm.releases.hashicorp.com"
  version          = "0.22.1"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [
    templatefile("${path.module}/files/values.yaml",
      {
        IAM_ROLE_ARN = module.irsa.arn
        KMS_KEY_ID   = aws_kms_key.vault.key_id
    })
  ]
  set_sensitive {
    name = "server.extraEnvironmentVars.GOOGLE_APPLICATION_CREDENTIALS_BASE64_ENCODED"
    value = var.google_application_credentials_vault_server_base64
  }
}

module "irsa" {
  source         = "Young-ook/eks/aws//modules/iam-role-for-serviceaccount"
  version        = "1.7.5"
  name           = "vault-server"
  namespace      = "vault"
  serviceaccount = "vault"
  oidc_url       = var.oidc_provider
  oidc_arn       = "arn:aws:iam::${var.account_id}:oidc-provider/${var.oidc_provider}"
  policy_arns    = []
}

resource "aws_iam_role_policy" "assume_role_for_irsa" {
  name = "assume-role"
  role = module.irsa.name
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

resource "aws_kms_key" "vault" {
  policy = jsonencode(
    {
      "Id" : "key-consolepolicy-3",
      "Version" : "2012-10-17",
      "Statement" : [
        {
          "Sid" : "Enable IAM User Permissions",
          "Effect" : "Allow",
          "Principal" : {
            "AWS" : "arn:aws:iam::${var.account_id}:root"
          },
          "Action" : "kms:*",
          "Resource" : "*"
        },
        {
          "Sid" : "Allow access for Key Administrators",
          "Effect" : "Allow",
          "Principal" : {
            // FIXME: Seems a bit hardcoded 
            "AWS" : "arn:aws:iam::${var.account_id}:role/aws-reserved/sso.amazonaws.com/${var.primary_region}/AWSReservedSSO_AdministratorAccess_6f030177475b118d"
          },
          "Action" : [
            "kms:Create*",
            "kms:Describe*",
            "kms:Enable*",
            "kms:List*",
            "kms:Put*",
            "kms:Update*",
            "kms:Revoke*",
            "kms:Disable*",
            "kms:Get*",
            "kms:Delete*",
            "kms:TagResource",
            "kms:UntagResource"
          ],
          "Resource" : "*"
        },
        {
          "Sid" : "Allow use of the key",
          "Effect" : "Allow",
          "Principal" : {
            "AWS" : "${module.irsa.arn}"
          },
          "Action" : [
            "kms:Encrypt",
            "kms:Decrypt",
            "kms:ReEncrypt*",
            "kms:GenerateDataKey*",
            "kms:DescribeKey"
          ],
          "Resource" : "*"
        },
        {
          "Sid" : "Allow attachment of persistent resources",
          "Effect" : "Allow",
          "Principal" : {
            "AWS" : "${module.irsa.arn}"
          },
          "Action" : [
            "kms:CreateGrant",
            "kms:ListGrants",
            "kms:RevokeGrant"
          ],
          "Resource" : "*",
          "Condition" : {
            "Bool" : {
              "kms:GrantIsForAWSResource" : "true"
            }
          }
        }
      ]
    }
  )
}

resource "aws_kms_alias" "vault" {
  name          = "alias/Vault"
  target_key_id = aws_kms_key.vault.key_id
}


resource "kubernetes_ingress_v1" "vault_ui_ingress" {
  metadata {
    name = "vault-ui"
    namespace = "vault"
  }
  spec {
    ingress_class_name = "nginx"
    rule {
      host = "vault.private.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "vault-ui"
            port {
              number = 8200
            }
          }
          }
          path = "/"
        }
      }
    }
    # Remove this after validation that it doesn't break anything.
    rule {
      host = "vault.voxelplatform.com"
      http {
        path {
          backend {
            service {
              name = "vault-ui"
            port {
              number = 8200
            }
          }
          }
          path = "/"
        }
      }
    }
}
}
