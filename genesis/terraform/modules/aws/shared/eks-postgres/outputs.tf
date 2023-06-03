output "db_instance_name" {
  value = var.db_identifier
}
output "db_instance_username" {
  value = "postgres"
}
output "db_instance_password" {
  value = random_password.password.result
  sensitive = true
}
output "db_instance_host" {
  value = "${helm_release.primary.name}-postgresql-ha-pgpool.${helm_release.primary.namespace}.svc.cluster.local"
}
output "db_instance_port" {
  value = "5432"
}