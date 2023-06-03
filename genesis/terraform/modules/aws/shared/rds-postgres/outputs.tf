output "db_instance_endpoint" {
  value = module.primary.db_instance_endpoint
}
output "db_instance_name" {
  value = module.primary.db_instance_name
}
output "db_instance_username" {
  value     = module.primary.db_instance_username
  sensitive = true
}
output "db_instance_password" {
  value     = module.primary.db_instance_password
  sensitive = true
}

output "db_instance_address" {
  value = module.primary.db_instance_address
}

output "db_instance_port" {
  value = module.primary.db_instance_port
}
