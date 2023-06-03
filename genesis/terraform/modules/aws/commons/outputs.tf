
output "grafana_url" {
  value = var.grafana_url
}

output "grafana_api_key" {
  sensitive = true
  value     = var.grafana_api_key
}

output "grafana_irsa_arn" {
  value = var.grafana_irsa_arn
}

output "root_account_id" {
  value = var.root_account_id
}
output "platform_account_id" {
  value = var.platform_account_id
}
output "production_account_id" {
  value = var.production_account_id
}
output "staging_account_id" {
  value = var.staging_account_id
}
output "development_account_id" {
  value = var.development_account_id
}

output "prime_account_id" {
  value = var.prime_account_id
}

output "galileo_account_id" {
  value = var.galileo_account_id
}

output accounts {
  value = {
    root = {
      id = var.root_account_id
    }
    prime = {
      id = var.prime_account_id
    }
    development = {
      id = var.development_account_id
    }

    staging = {
      id = var.staging_account_id
    }

    platform = {
      id = var.platform_account_id
    }

    production = {
      id = var.production_account_id
    }

    galileo = {
      id = var.galileo_account_id
    }
  }
}

output "aws_notifications_slack_app_webhook_url" {
  value = var.aws_notifications_slack_app_webhook_url
  sensitive = true
}

output services_root_ca {
  value = {
    cert_pem = base64decode(var.voxel_services_root_cert_base64encoded)
    key_pem = base64decode(var.voxel_services_root_key_base64encoded)
    password = base64decode(var.voxel_services_root_password_base64encoded)
  }
  sensitive = true
}
