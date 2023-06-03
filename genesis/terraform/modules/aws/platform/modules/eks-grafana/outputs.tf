output "grafana_url" {
  value = "https://${var.domain}"
}

output "grafana_api_key" {
  value     = grafana_api_key.tf.key
  sensitive = true
}

output "grafana_irsa_arn" {
  value = module.grafana_irsa.arn
}

