output "db_instance_name" {
  value = var.db_identifier
}

output "db_instance_password" {
  value = random_password.password.result
  sensitive = true
}
