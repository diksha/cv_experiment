resource "random_password" "db_password" {
  length           = 20
  special          = true
  override_special = "_-@"
}

resource "aws_iam_user" "smtp_user" {
  name = "snipeit_smtp_user"
}

resource "aws_iam_access_key" "smtp_user" {
  user = aws_iam_user.smtp_user.name
}

data "aws_iam_policy_document" "ses_sender" {
  statement {
    actions   = ["ses:SendRawEmail"]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "ses_sender" {
  name        = "snipeit_ses_sender"
  description = "Allows sending of e-mails via Simple Email Service"
  policy      = data.aws_iam_policy_document.ses_sender.json
}

resource "aws_iam_user_policy_attachment" "ses_attach" {
  user       = aws_iam_user.smtp_user.name
  policy_arn = aws_iam_policy.ses_sender.arn
}

resource "aws_ses_email_identity" "it" {
  email = "it@voxelai.com"
}


resource "helm_release" "main" {
  name             = "snipeit"
  namespace        = "snipeit"
  chart            = "snipeit"
  repository       = "https://storage.googleapis.com/t3n-helm-charts"
  version          = "3.4.0"
  wait             = true
  timeout          = 300
  create_namespace = true
  values = [templatefile("${path.module}/files/service-helm-values.yaml", {
    })
  ]
  set_sensitive {
    name = "config.snipeit.key"
    value = var.snipeit_key
  }

  set_sensitive {
    name = "mysql.mysqlPassword"
    value = random_password.db_password.result
  }

  set_sensitive {
    name = "config.snipeit.envConfig.MAIL_USERNAME"
    value = aws_iam_user.smtp_user.name
  }

  set_sensitive {
    name = "config.snipeit.envConfig.MAIL_PASSWORD"
    value = aws_iam_access_key.smtp_user.ses_smtp_password_v4
  }
}
