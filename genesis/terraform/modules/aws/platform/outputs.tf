output "account_id" {
  value = var.target_account_id
}
output "argo_admin_username" {
  value = module.argo-cd.admin_username
}

output "argo_admin_password" {
  value     = module.argo-cd.admin_password
  sensitive = true
}

output "grafana_url" {
  value = "https://grafana.${var.root_domain}"
}

output "grafana_api_key" {
  value     = module.grafana.grafana_api_key
  sensitive = true
}

output "grafana_irsa_arn" {
  value     = module.grafana.grafana_irsa_arn
  sensitive = true
}

output "primary_vpc_id" {
  value = module.vpc.vpc_id
}

output "sentry_password" {
  value     = module.sentry.admin_password
  sensitive = true
}

output "metaverse_password" {
  value = module.metaverse.db_instance_password
  sensitive = true
}

output "metaverse_internal_password" {
  value = module.metaverse_internal.db_instance_password
  sensitive = true
}

output "retool_shared_db_password" {
  value = module.retool.shared_db_password
  sensitive = true
}